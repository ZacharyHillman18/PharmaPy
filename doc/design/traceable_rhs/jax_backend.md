# JAX and CasADi backends: what was built, controllers with memory, and speed

*Branch `JAXBackend`, 2026-10-21. Follows `casadi_vs_jax.md` (the decision
note) and implements its recommendation: one refactor against JAX, with
CasADi reached through the JAX-compiled model. Measured with JAX 0.11.2 and
CasADi 3.8.1 on CPU, double precision, on this laptop; wall times on it vary
by about +-20% from run to run, so compare within a table, not across them.*

## 1. What exists now

| Piece | Where | What it does |
|---|---|---|
| `xp` | `PharmaPy/xp.py` | Imported as `np` on the RHS path. Outside a trace its names *are* numpy's functions, so numeric runs pay nothing. While a trace is built (`with xp.tracing_at(t):`) they dispatch tracers to `jax.numpy`. Adds `set_at` / `add_at` (in place on numpy, a new array when traced), `is_traced`, `as_float`, `of_time`, `time_inputs`. Never imports JAX itself. |
| `TracedModel` | `PharmaPy/TracedBackends.py` | Compiles a unit's own `unit_model` once and serves the RHS, the exact Jacobian, exact Jacobian-vector products, d/dt, and the model as a `casadi.Function`. Functions of time a trace cannot follow (a profile written as plain Python, a recorded upstream trajectory) become *inputs* of the compiled function, evaluated on the number before each call. |
| `JaxBackend` | same | `ScipyBackend` (BDF by default) on the compiled RHS, with the exact Jacobian. Events, segmentation and the stall detector are ScipyBackend's own. |
| `JaxAssimuloBackend` | same | `AssimuloBackend` (CVode) on the compiled RHS: exact dense Jacobian below 100 states, exact Jacobian-vector products under SPGMR from 100 up when the unit asks for Krylov (crystallizers do). Events as under AssimuloBackend. |
| `CasadiBackend` | same | CasADi's CVODES on the compiled model through a `casadi.Callback` with exact Jacobian and exact forward derivatives: dense CSparse LU below 100 states, GMRES on exact Jacobian-vector products above. A unit with algebraic states goes to IDAS, so a ConcurrentVessel recycle (`tear_mode='algebraic'`) runs on it. |

All are importable as `from PharmaPy.IntegratorBackends import JaxBackend`
(and the others); nothing imports JAX or CasADi until one is constructed.
Install with `pip install jax` or `pip install jax casadi` (extras `[jax]` and
`[casadi]` in `pyproject.toml`). A dry run into `pharmNew` adds jax/jaxlib
0.12, casadi 3.8.1, ml_dtypes and opt_einsum, and leaves numpy and scipy as
they are.

```python
from PharmaPy.IntegratorBackends import CasadiBackend, JaxAssimuloBackend

cryst = ContinuousCrystallizer(integrator=JaxAssimuloBackend(options={'maxh': 60}), ...)
cryst.solve_unit(runtime=3600)          # same results object, same replay

cv = ConcurrentVessel(members, links, integrator=CasadiBackend(),
                      tear_mode='algebraic')   # recycle as a DAE on IDAS
cv.solve_unit(runtime=3600)
```

**Covered and checked against the numeric path** (compiled RHS equal to the
unit's own to 1e-14 to 1e-12; solutions equal to AssimuloBackend's to about
1e-6, i.e. to integration tolerance): batch, semibatch and continuous reactors;
batch, semibatch and continuous crystallizers, FVM and moments; temperature
profiles and energy balances; ConcurrentVessel acyclic, with a cross-unit
controller, and with a recycle as algebraic tears (CasADi IDAS against
AssimuloDAEBackend: 1.3e-8). `tests/test_traced_backends.py`, 20 tests.

**Not covered:** the filter / deliquoring units (`SolidLiquidSep_Refactored`,
which also end on terminal events), ConcurrentVessel recycles iterated inside
the RHS (refused, with a pointer to `tear_mode='algebraic'`), and parameter
sensitivities.

### What changed in existing code

The vessel's structure, interfaces and assembly order are unchanged. The
edits are the patterns `casadi_vs_jax.md` listed, now applied over the
crystallizer and ConcurrentVessel paths too:

| Pattern | Example | Sites |
|---|---|---|
| `x[i] = v` / `x[i] += v` | `moments = np.set_at(moments, n, ...)` | moments, limiter, flux limiter, species transfer, every balance term, rate packing, thermo, ConcurrentVessel rates |
| `if` on a value | `theta = np.where(growth > 0, upwind_a, upwind_b)` | FVM upwind ratio, growth/dissolution switch, MechanismView direction, set_mass rescaling, flow terms on an empty vessel, soft floor, limiter short cuts |
| value-dependent mask | ratio for every species, `inf` where not short | reaction positivity guard, vessel limiter |
| `float(x)`, `max(x, 0)` | `np.as_float`, `np.maximum` | slurry volume, solid fraction, moments mass transfer, outlet values, tear unpacking |
| validation | `if not np.is_traced(v) and ...` | mass fraction sum |
| functions of time | `np.of_time(self.temp_func, time)` | temperature profiles, recorded upstream trajectories |

Where a numeric branch and its traced counterpart differ in cost or in types,
the numeric branch is kept as it was and the traced one added beside it, so
numeric callers run the same code.

**Numeric results.** Every regression check is unchanged: 156 of 172 dumped
arrays bit-identical on scipy and on Assimulo, FlowsheetTester's stages
identical on both backends (stage 2 still stalls on scipy, as on HEAD; on
scipy stage 10 reports the batch moments crystallizer's mass drift as 6.0e-13
for 4.5e-13), Tester and crystTester identical, pytest 118 passed (without
JAX installed, the 18 traced tests skip). The 16 arrays that moved are all the
moments crystallizer's, from fix 1 below.

**Two things the trace found that were wrong numerically as well:**

1. **A population balance counted twice at a net rate of exactly zero.**
   `MechanismView` kept the crystallization result on the forward view when
   the net mass rate is `>= 0` and on the reverse view when it is `<= 0`, so
   at exactly zero both kept it. A moments crystallizer starts there (nucleus
   size 0, mu_2 = 0), and mu_0 grew at twice its rate at that one point. The
   original `method_of_moments` evaluates once. The forward view now owns
   zero. The moments crystallizer's final state moves by at most 4e-8
   (mu_n), 2e-6 (outlet flow) and 6e-5 (supersaturation); nothing else moves.
2. **A silent fallback.** `_slurry_volume` swallowed `TypeError` and fell back
   to a cached volume, which under a trace would have frozen the slurry
   volume into the compiled function. It now re-raises when traced, which is
   how the `float()` in `solid_volume_fraction` was found.

### Rules for code users write (controllers, kinetics, mechanisms)

A traced backend runs user code through the same tracer, and it fails loudly
at the first line that breaks a rule, never silently:

- arrays are written with `np.set_at` / `np.add_at` (`from PharmaPy import xp as np`);
- no `if` on a computed value, no `float(x)`, no builtin `max`/`min` on values
  (`np.where`, `np.maximum`);
- functions of time alone may be any Python. Written with `xp`
  (`np.where(t < 100, np.interp(t, ...), 273.15)`) they are compiled in;
  written with plain numpy or an `if`, they are passed in as inputs, which
  costs the function's own run time on every call.

The cross-unit controller in `tests/test_concurrent_vessel.py` uses `float()`
and `max()`; the same controller written to these rules is in
`tests/test_traced_backends.py` and runs on JAX and CasADi.

## 2. Controllers with memory (PID)

### Why memory cannot live on the controller object

Every backend calls the RHS at points that are not on the trajectory: Newton
iterates, difference-quotient columns, rejected steps, event root-finding,
the replay. A controller that accumulates `self.integral += error * dt` inside
`actuate` integrates over all of those calls. It is therefore wrong already
under AssimuloBackend and ScipyBackend, and the replay cannot reproduce it.
Under a trace it is worse: the accumulator is read once, at trace time, and
frozen into the compiled function. The rule "controllers own no solver
states" was right for stateless control laws, but a PID's memory has to
become solver states.

### The design: controllers declare solver states, as mechanisms do

```python
class PIDController(Controller):
    """u = u0 + Kp*e + Ki*I - Kd*dy_f/dt, clipped to [u_min, u_max].

    I (the integral) and y_f (the filtered measurement) are solver states,
    so they are integrated on the accepted trajectory, replayed exactly,
    and traced like every other state.
    """

    solver_states = (
        StateVariable("pid_integral", dim=1, state_type="diff"),
        StateVariable("pid_filtered", dim=1, state_type="diff"),
    )

    def output(self, time, completed_state, unit):
        y = self.measure(unit)                       # e.g. a downstream [C]
        e = self.setpoint(time) - y
        integral = completed_state[StateKey("pid_integral")]
        filtered = completed_state[StateKey("pid_filtered")]
        raw = (self.u0 + self.Kp * e + self.Ki * integral
               - self.Kd * (y - filtered) / self.tau_f)
        return raw, np.clip(raw, self.u_min, self.u_max), e, y, filtered

    def get_solver_state_rates(self, time, completed_state, unit):
        raw, u, e, y, filtered = self.output(time, completed_state, unit)
        return {
            # back-calculation anti-windup: the integral stops growing
            # while the output is held at a limit
            StateKey("pid_integral"): e + (u - raw) / self.tracking_time,
            StateKey("pid_filtered"): (y - filtered) / self.tau_f,
        }

    def actuate(self, time, completed_state, unit, resolved_inlets=None,
                resolved_outlets=None):
        _, u, *_ = self.output(time, completed_state, unit)
        self.operating_conditions[self.target_key] = u
```

What the vessel needs, about 40 lines in `MultiPhaseVessel`:

- `compile_structure` adds the controller's `solver_states` to the solver
  state collection as global keys (no phase), as `global_temp` already is;
- `unit_model` asks the controller for `get_solver_state_rates` and packs
  them through the existing `global_rates` path of `pack_state_rates`;
- `create_solver_init_states` takes their initial values from the controller
  (`I0`, and `y_f0` = the first measurement).

Consequences:

- **Every backend handles it unchanged**, numeric and traced: it is two more
  states. The replay reproduces the controller exactly.
- **Derivative action** acts on a filtered measurement (one more state), never
  on `de/dt`: a setpoint step would kick it, and `dy/dt` of a state would mean
  evaluating the RHS inside the RHS.
- **Anti-windup** by back-calculation is `np.clip` plus one term, smooth away
  from the limits and traceable.
- **ConcurrentVessel:** a PID on R01 measuring R02 is two states in R01's
  block; the coordinator already concatenates member states. Cascade stays
  PULL: the inner loop reads the master's output, and the master's integral
  is a state of the master's vessel.
- **Single-controller rule:** unchanged. Several loops on one vessel are one
  controller object with several states (a MIMO controller), not several
  controllers.
- **Discrete (sampled) PID** is a different object: the memory changes only
  at sample times, which makes the model hybrid. Assimulo (time events) and
  the scipy segmentation can carry it; under JAX or CasADi it belongs outside
  the compiled model, integrating one sample interval at a time with `u`
  held, which is also how an MPC loop is structured.
- **For MPC** the PID is replaced by the optimizer. The next step there is
  exposing manipulated variables as inputs `u` of the compiled function
  (`f(t, x, u)`, keyed by `OperatingKey`). `xp.time_inputs` already passes
  values into the compiled function from outside; the same path carries `u`.

This reopens the earlier "controllers own no solver states" decision, which
is why it is a design here and not code. *It was implemented afterwards, on
branch `ControllerSolverStates`; see `../controller_solver_states.md`.*

## 3. Is it faster than finite differences, and by how much

### Per call

One evaluation at a mid-run state (900 s into the run), median of the
measured runs. A difference-quotient Jacobian costs n+1 numeric evaluations;
a difference Jacobian-vector product costs one.

| Unit | states | numeric RHS | compiled RHS | FD Jacobian | exact Jacobian | FD J*v | exact J*v | compile |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| continuous crystallizer, 35 bins | 40 | 3.4 ms | 0.11 ms | 140 ms | 0.57 ms | 3.4 ms | 0.18 ms | 4.4 s |
| continuous crystallizer, 100 bins | 105 | 3-5 ms | 0.11 ms | 300-500 ms | 3.2 ms | 3-5 ms | 0.23 ms | 3.6 s |
| continuous crystallizer, 200 bins | 205 | 3-4 ms | 0.23 ms | 800 ms | 19 ms | 3-4 ms | 0.36 ms | 3.2 s |
| batch crystallizer, 35 bins (profile as input) | 40 | 1.5 ms | 0.20 ms | 60 ms | 0.65 ms | 1.5 ms | 0.32 ms | 2.1 s |
| moments crystallizer | 9 | 3 ms | 0.07 ms | 30 ms | 0.25 ms | 3 ms | 0.17 ms | 3.1 s |
| continuous reactor | 5 | 1.1 ms | 0.08 ms | 7 ms | 0.12 ms | 1.1 ms | 0.11 ms | 1.1 s |

So per call the compiled right-hand side is 7 to 40 times cheaper, an exact
Jacobian 40 to 250 times cheaper than its difference quotient, and an exact
Jacobian-vector product 5 to 20 times cheaper.

### End to end

Integration time (solver plus model calls plus events) and total time (adds
compiling and the replay), against the default `AssimuloBackend` (CVode,
Krylov with difference-quotient products). Final round, all optimisations in.

| Unit | default CVode | JaxBackend (scipy BDF) | JaxAssimuloBackend | CasadiBackend |
|---|---:|---:|---:|---:|
| batch crystallizer, 35 bins | 7.8 s (10.8) | 3.6 s, 2.2x (12.0) | 3.1 s, 2.5x (10.9) | 1.1 s, 7.1x (3.7) |
| continuous crystallizer, 35 bins | 16-21 s (23-31) | 4.5 s, 4x (16.9) | 4.5 s, 4x (17.4) | 1.1 s, 15-20x (5.7) |
| moments crystallizer | 3.0 s (4.1) | 2.3 s, 1.3x (8.1) | 0.7 s, 4.6x (5.1) | 0.3 s, 9x (3.8) |
| batch crystallizer, 200 bins | 31.7 s (46.6) | 23.4 s, 1.4x (41.1) | 14.0 s, 2.3x (32.9) | 16.3 s, 1.9x (17.6) |
| continuous crystallizer, 200 bins | 65-78 s (88-105) | 26.6 s, 2.9x (54.9) | 14.9 s, 5x (50.4) | 12.3 s, 6x (14.7) |
| your `crystTester` case (199 bins) | 11.6 s (18.3) | 6.8 s, 1.7x (15.3) | 3.8 s, 3.1x (9.9) | 4.7 s, 2.5x (5.7) |
| continuous reactor | 0.2 s (0.3) | 0.1 s (1.1) | 0.1 s (1.2) | 0.04 s (1.3) |

Total times in parentheses. scipy's own BDF with difference quotients is the
slowest everywhere (0.1x to 0.7x of CVode), which is where JaxBackend gains
most: 3 to 11 times on the crystallizers.

### Reading it

- **Crystallizers: yes, by 2 to 6 times on CVode and 2 to 20 on CasADi**,
  integration only. Exact derivatives remove the difference quotients (n
  evaluations per Jacobian, one per Krylov iteration), and the compiled RHS
  makes each remaining evaluation 7 to 40 times cheaper.
- **Reactors: no**, as expected. They integrate in 0.3 s, and compiling costs
  1 s. The gain there would come only from compiling once and solving many
  times (parameter estimation, MPC), which needs the compiled model cached
  across solves; it is rebuilt on every `solve_unit` today.
- **What is left is not the integration.** The numeric replay walks the
  Python objects once per output point, 3 to 5 ms a point. CVode reports
  every internal step (350 to 7,000 of them here), so the replay takes 1 to
  35 s, now often more than the integration. CasADi reports on a 201-point
  grid, which is most of why its totals are so low. Passing `time_grid` to
  `solve_unit` gives CVode the same saving. Events are evaluated numerically
  too, at about 1-2 ms each, once per step: 2 to 14 s of the integration
  times above.
- **Which backend.** CasADi is fastest for a single solve, and its event
  location is off by default (it steps through the regime switches under its
  error control). JaxAssimuloBackend is the closest drop-in for
  AssimuloBackend: same events, same reported points, 2 to 5 times faster
  integration. JaxBackend matters where only scipy is available.
- **Numeric runs cost 2-7% more than on HEAD.** With the dispatcher installed
  only while a trace is built, `np.where` on a numeric run is numpy's own
  function; what remains is the cheap checks the traced branches need
  (`is_traced` answering False inline, `set_at` in place of `x[i] = v`,
  `where` in place of an `if`). One evaluation of the 35-bin continuous
  crystallizer makes 2,796 Python calls against HEAD's 2,634 (+6%), and
  crystTester, timed alternately with HEAD, ran 21.4-22.0 s on HEAD and
  23.1-23.4 s here. A first version dispatched every numpy call and cost
  20-30%; it is gone.
### Where more speed would come from

1. Replay through the compiled model: the outputs at all reported times in
   one vectorised call instead of a Python walk per point.
2. Events through the compiled model rather than the numeric unit.
3. A sparse Jacobian for large units (graph colouring: about ten JVPs instead
   of 205 for the 200-bin crystallizer), and giving CasADi its sparsity.
4. Keeping the compiled model across solves of the same unit.
