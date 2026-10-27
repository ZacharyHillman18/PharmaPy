# Controllers with solver states, PI(D) control, and phase connections

*Branch `ControllerSolverStates`, off `JAXBackend` (a07e67b), 2026-10-27.*

## 1. Controllers declare solver states

A controller may now declare solver states, the way a mechanism does:

```python
class Controller:
    solver_states = ()            # StateVariable templates

    def get_solver_state_rates(self, time, completed_state, unit,
                               resolved_inlets=None, resolved_outlets=None,
                               operating_conditions=None): ...   # {StateKey: rate}

    def get_solver_state_residuals(self, ...): ...              # {StateKey: residual}, for 'alg'

    def initial_solver_state(self, state_key, unit=None): ...   # default: the attribute of that name
```

The vessel adds them to its solver states as global keys (no phase), the way
it adds `global_temp`. It asks the controller for their rates after the balances,
handing it the streams and operating conditions of the same evaluation (what was
actually applied, after the floor and the positivity limiter), and packs them
with the other global rates. Residuals of states declared `state_type='alg'`
join the mechanisms' residuals. The controller reads its states from
`completed_state`, like everything else. After every evaluation the vessel copies
their values back onto the controller (`update_solver_states`), as it does for
mechanisms.

About 60 lines in `MultiPhaseVessel` and 90 in `Controller`. Nothing else changed:

| Question | Answer |
|---|---|
| Does it work in a ConcurrentVessel? | Yes, with no change there. A member's solver states, the controller's included, are a block of the combined vector. The state lives in the member that owns the controller, even when it measures another member: in the tests a PI loop on R01's feed integrates R02's error as a state of R01. Cascade stays PULL; a master loop's integral is a state of the master's vessel. |
| Was it only a matter of allowing states? | Mostly. Two more things were needed: the rates had to be computed with the streams of the same evaluation (for anti-windup against what was applied), and the starting value of a state has to be available before the first evaluation (`initial_solver_state(key, unit)`; a filter starts at the measurement). The jacket loop also needed a manipulated variable the vessel did not have (section 2). |
| Does it make the controller traceable? | Yes. The memory is part of `y`, so the compiled function is still a pure function of `(t, y)`. A controller's own code must follow the PharmaPy.xp rules, as a mechanism's must. `tests/test_controller_states.py` traces a two-loop PID and a cross-unit PI on JAX and CasADi. |
| Replay | The states are in the trajectory, so the replay reproduces the controller exactly, and the results carry them (`result.level_integral`, `result.temperature_integral`, ...). |

### Do controllers need algebraic states and residuals? Yes, now they can

They are needed for a control law that is implicit: a manipulated variable
defined by a constraint it appears in, `g(x, u) = 0` with `dg/du` invertible
(index 1). Examples are:

- holding a ratio or a balance exactly, rather than with a stiff high-gain loop. The
  test case holds the vessel's mass, outflow by mass equal to inflow, with the
  outlet's volumetric flow as the algebraic state;
- a valve or pump characteristic solved for its position;
- ideal control of a measurement that depends algebraically on the manipulated
  variable within the same instant, such as a split fraction or a blend ratio.

They are *not* needed for PID, whose memory is differential. They are also not a way
to impose a differential state's value (`T = T_set` with the jacket temperature as
the unknown is index 2, because the jacket only appears in `dT/dt`; neither IDA nor
IDAS solves that).

A ConcurrentVessel did not have controller residuals before. Its algebraic
states were only its tear streams, which the coordinator owns. A member's
controller residuals now reach a ConcurrentVessel through the member's layout,
like any other member state. `test_algebraic_controller_state_inside_a_concurrent_vessel`
covers it. An algebraic state makes the vessel a DAE, so it needs a DAE backend:
AssimuloDAEBackend, DiffeqpyBackend, or CasadiBackend (IDAS). An ODE backend
refuses it, as before.

## 2. PI(D) controllers

`PIDLoop` holds the arithmetic of one loop: ISA form around a bias, direct or
reverse action, and an integral and filtered measurement as solver states. The
derivative acts on the filtered measurement, so there is no set-point kick and no
derivative of the RHS. Anti-windup is back-calculation against the applied output.

| Controller | Equivalent of | What it does |
|---|---|---|
| `PIDContinuousVesselVolume` | `DefaultContinuousVesselVolume` | PI(D) volume loop on the outlet, with the same feed-forward and no-backflow floor. Default Kc as the P loop (1 / a hundredth of a residence time), Ti = 4/Kc (critically damped for an integrating level). |
| `PIDContinuousVesselController` | `ContinuousVesselController` | Drop-in: temperature imposed on the profile exactly as before, volume by the PI(D) loop. |
| `PIDTemperatureController` | `SimpleTemperatureController` | Tracks the profile through the jacket with a PI(D) loop, keeping the energy balance. IMC defaults from the vessel's own jacket time constant: Ti = m cp / UA, Kc = 4. |
| `JacketPIDContinuousVesselController` | `ContinuousVesselController` | Both loops PI(D): temperature through the jacket, volume through the outlet. |

The jacket loop needed a handle on the jacket. `MultiPhaseVessel.UTILITY_TEMPERATURE`
(`OperatingKey('temp_in', port='utility')`) overrides `Utility.temp_in` for an
evaluation. The energy balance and the reported `q_ht` both read it. It belongs to
no stream, so stream resolution never sees it.

On the reacting CSTR of the tests, whose liquid densifies as it reacts:

- **Volume:** the proportional loop leaves the volume 4e-4 below target. That is
  (F_out - F_in)/Kc, since the outflow has to be about 4 % below the inflow. The PI
  loop holds it to 1e-8.
- **Jacket temperature:** the PI loop holds the set point against an exothermic
  reaction to 4e-4 K before a step and 1e-5 K after it. A proportional jacket loop
  is 0.14-0.18 K off.
- **Anti-windup:** after the jacket saturates, back-calculation cuts the overshoot
  from 1.15 K to 0.14 K.

## 3. Phase connections without kinetics; DirectTransfer

`PhaseConnection` no longer has a `kinetics` field. Nothing read it, the mechanism
owns its rate law, and requiring it made a connection whose mechanism needs no
kinetics impossible to build. `CrystKinetics` still reaches the population balance
through the mechanism (`connection.mechanism.mechanism_kinetics`).

`DirectTransfer` was unusable. It expected a `kinetics` argument the vessel never
passes and called a `get_rate` no kinetics class has. Because `compile_structure`
gave it to every connection without a mechanism, any such connection failed on
its first evaluation. It now owns its rate:

```python
DirectTransfer(rate)                 # kg/s source -> sink: callable(source, sink, t, state) or constant
DirectTransfer.first_order(k, species=['C'])   # k * source mass of each moving species
DirectTransfer(rate, heat_of_transfer=-2e5)    # J/kg moved, into the energy balance
```

An array rate is per species, and a negative entry flows back. A scalar is split by
the connection's `species_weights`, or else by the source's composition. The
vessel's positivity limiter throttles it like any other transfer. A connection
with no mechanism is now refused at compile time with a message saying so.
`tests/test_direct_transfer.py` checks:

- the exact first-order solution;
- conservation;
- both splits;
- reverse flow;
- the limiter;
- the heat term;
- the refusal;
- a JAX trace.

## 4. Result_Figures.ipynb cases A1-A6 on the traced backends (a curiosity)

The notebook's A1-A6 runs were timed with its own builders, grids and `maxh`,
the notebook left unchanged, on legacy PharmaPy, the refactored vessels on
AssimuloBackend (as in the notebook), and the refactored vessels on the traced
backends. Times are build plus solve (the notebook's `timed`), best of 3, after
one warm-up compile in the process. Two runs on this laptop, whose speed drifts
by up to 3x between runs, so compare along a row:

| Case | legacy | CVode (refactored) | JaxAssimulo | CasADi | of which compiling |
|---|---:|---:|---:|---:|---:|
| A1-A3 reactors, CSTR | 0.006-0.03 s | 0.03-0.36 s | 0.24-0.9 s | 0.24-0.8 s | 0.2-0.7 s |
| A4, A5 batch crystallizers | 0.48-1.8 s | 1.5-4.7 s | 0.9-2.2 s | 0.93-2.0 s | 0.6-1.4 s |
| A6 MSMPR, run 1 | 1.20-1.34 s | 7.9-8.6 s | 2.4-2.6 s | 1.52-1.75 s | 1.1-1.4 s |
| A6 MSMPR, run 2 | 3.0-3.3 s | 16.6-20.9 s | 3.8-4.7 s | 2.44-2.99 s | 1.5-1.8 s |

- **With compiling included, the traced backends do not reliably beat legacy.**
  On the MSMPRs CasADi comes out about even: 15-30 % slower in one run, 10-20 %
  faster in the other. The batch crystallizers are 1.1-2 times slower, and the
  reactors 20-40 times. A compile of 0.2-1.4 s is longer than the legacy solve
  in every case.
- **Without compiling, the fastest traced run beats legacy on every
  crystallizer case but one** (A5 p=0.5 in run 1, 0.55 s against 0.48 s), by
  up to 6x on the MSMPRs. These figures are approximate: total minus the
  measured compile time. Against the refactored code on CVode, CasADi
  is 1.0-7x faster including the compile (the MSMPRs 5-7x). For many solves of
  one model (parameter estimation, MPC) the compile is paid once, which is the
  case where keeping the compiled model across solves would pay off.
- The script is `timing_A1_A6.py` in the session scratchpad. It found a bug: an
  empty `options={}` gave JaxAssimuloBackend `maxh=1` (21,600 steps on the
  six-hour CSTR). It is fixed on this branch and covered by a test.
