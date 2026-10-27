# Making PharmaPy's right-hand side traceable: CasADi, JAX, or both

*Design note, 2026-10-16. Measured on `megaVessel` at `9bb573d` with CasADi 3.8.1
and JAX 0.11.2 (CPU, x64). Everything below was run, not assumed; the patches
and scripts are in this folder.*

## Summary

- **The integrator backends are the small part.** A CasADi or JAX backend is a
  new class next to the existing ones; nothing in the existing backends changes.
  What has to change is the code the right-hand side (RHS) runs through: the
  vessel's assembly, the phases and thermo, kinetics, mechanisms and controllers.
- **The structure does not change.** The object model (phases as objects,
  `update_phases_from_state`), the `StateCollection` layout, the mechanism and
  controller interfaces and the vessel's assembly order all trace as they are.
  What changes is about seven *code patterns* inside them: in-place array
  writes, `if` on a computed value, and so on (section 3).
- **One refactor serves both.** Route array calls through a small namespace
  module (`xp`, imported as `np`) and remove those patterns. On the continuous
  reactor, isothermal and with an energy balance, that took 105 added and 124
  removed lines across 10 existing modules, plus a new 63-line `xp.py`. The RHS
  then JIT-compiles under JAX and matches numpy to round-off (4.7e-13 at the start
  state, 1e-16 to 3e-15 elsewhere), and its exact Jacobian matches central
  differences to 8e-9. Numeric solves stayed **bit-identical**.
- **CasADi gets the same model two ways.**
  1. **Through JAX (tested):** wrap the JAX-compiled RHS and its exact Jacobian
     in a `casadi.Callback`. CasADi's own CVODES then integrated the reactor for
     1 h in 0.6 s and agreed with scipy to 3e-9. No CasADi-specific code is
     needed in PharmaPy.
  2. **Natively (probed, not finished):** teach `xp` to emulate numpy's array
     API on CasADi's `SX`. This works but needs extra adapters JAX did not:
     1-D vs 2-D semantics, array methods and attributes. CasADi also fails
     *silently*: `float(symbol)` and writing a symbol into a numpy buffer both
     give NaN with no error.
- **Recommendation:** do the refactor against JAX first, since it fails loudly
  at the exact line, and give CasADi the model through the Callback bridge.
  Build the native CasADi adapter only if MPC needs what the bridge cannot give:
  CasADi-side sparsity detection, C code generation, or second derivatives
  inside CasADi.

## 1. What a tracer is, and why PharmaPy breaks on it

CasADi and JAX both run the RHS once with stand-in values (a CasADi `SX`
symbol, a JAX tracer) and record the operations into a graph. From that graph
they get exact derivatives, compiled code and, for CasADi, optimal-control
transcriptions. The rules are the same for both:

- every number must come from the inputs, through operations the framework knows;
- Python must never need the *value* of a stand-in: no `if x > 0`, `float(x)`,
  `max(x, 0)`, or data-dependent shapes;
- for JAX, arrays are immutable (`x[i] = v` is an error); for CasADi, numpy
  buffers cannot hold symbols.

Python objects *may* be mutated during the trace. Phases can hold traced values
for the duration of one trace, which is why the object model survives.

## 2. Where it breaks today (probe on the unmodified code)

`probe_trace.py` feeds a CasADi symbol and a JAX tracer into each layer of a
real continuous reactor:

| Layer | CasADi `SX` | JAX tracer |
|---|---|---|
| numpy ufuncs on the value (`np.exp`, `np.maximum`, `np.where`, `np.sum`) | **work** (CasADi 3.8 dispatches them) | fail: numpy cannot hold a tracer |
| `if x > 0`, builtin `max(x, 0)` | fail | fail |
| `float(x)`, `numpy_buffer[i] = x` | **silently NaN** | fail |
| `RxnKinetics.get_rxn_rates` | fails: `np.dot` on SX's 2-D column shape (Kinetics.py:503) | fails: `np.asarray(temp)` (Kinetics.py:434) |
| `LiquidPhase.updatePhase(mass_j=...)` | fails: `np.asarray(value, dtype=float)` (Phases_Refactored.py:253) | same line |
| `DefaultContinuousVesselVolume.soft_floor` | fails: `if value >= width` (ProcessControl_Refactored.py:240) | same |
| whole `unit_model` | first failure in the phase setter | same |

Swapping numpy for a namespace that sends tracers to `jax.numpy`
(`probe_shim.py`) clears the numpy failures and leaves only the *logic* ones
in section 3.

## 3. What has to change, with examples

### 3.1 Array calls go through a namespace (mechanical, 1 line per module)

```python
# before                                   # after
import numpy as np                         from PharmaPy import xp as np
```

`xp` (63 lines in the spike) is numpy, except that a call whose arguments
include a JAX tracer goes to `jax.numpy`. A CasADi symbol goes to an adapter
table. Numeric callers get plain numpy, so numeric results do not move.
This is the only change that differs between the two frameworks.

### 3.2 In-place array writes return a new array (18 sites on the reactor path)

JAX arrays are immutable, and a numpy buffer cannot hold a tracer or a symbol.
`xp.set_at` / `xp.add_at` write in place on numpy, exactly as before, and
return a new array when traced:

```python
# Mechanisms.py, ReactionMechanism
species_massPerVol_rates = np.zeros(phase.num_species)
species_massPerVol_rates[mask] = species_rates                   # before
species_massPerVol_rates = np.set_at(species_massPerVol_rates,   # after
                                     mask, species_rates)

# MultiPhaseVessel.py, every balance term (7 sites, converted by a regex)
buffer.contributions[buffer.INLET, material_slice] += transfer.species_flow
buffer.contributions = np.add_at(buffer.contributions,
                                 (buffer.INLET, material_slice), transfer.species_flow)
```

Writes through a view need care: `row = buffer.contributions[OUTLET]; row[s] += v`
mutates the buffer in numpy but only a copy in JAX, so it must become
`add_at` on the buffer itself. The persistent buffers (`_solver_rate_buffer`,
`MaterialContributionBuffer`, `_inventory_buffer`) keep their numpy form for
numeric calls; `reset` restores it after a trace.

### 3.3 `if` on a computed value becomes `where` (7 sites)

```python
# Phases_Refactored.py, mass_j setter: before
total_mass = value.sum()
if total_mass > 0:
    self.mass = total_mass
    self.mass_frac = value / total_mass
else:
    self.mass = 0.0          # keep the last composition

# after
positive = total_mass > 0
safe_total = np.where(positive, total_mass, 1.0)      # no 0/0 in the unused branch
self.mass = np.where(positive, total_mass, 0.0)
self.mass_frac = np.where(positive, value / safe_total, self.mass_frac)
```

Both branches are always evaluated, so **both must be finite everywhere**. The
"safe" inner `where` is not decoration: a NaN in the unused branch poisons
JAX gradients, and CasADi's `if_else` evaluates both branches by default. That
matters most for regime switches such as growth against dissolution, where one
branch is a fractional power of a supersaturation that is negative in the
other regime:

```python
# Kinetics.py, CrystKinetics: picks which mechanisms to evaluate
if sup_sat >= 0:  subset_mech = ('nucl_prim', 'nucl_sec', 'growth')
else:             subset_mech = ('dissolution',)
# -> evaluate both on clamped inputs, then select
growing = sup_sat >= 0
growth  = growth_rate(np.where(growing, sup_sat, 0.0))
dissol  = dissolution_rate(np.where(growing, 0.0, sup_sat))
rate    = np.where(growing, growth, -dissol)
```

The same applies to `DefaultContinuousVesselVolume.soft_floor` and `get_gain`,
`compute_requested_phase_outlet_flow`, `MixedPhase.__getattr__`, and the
FVM's upwind choice (`if growth > 0:` in `OneDFVMMechanism`).

### 3.4 Value-dependent masks become fixed shapes (2 sites)

A boolean mask taken from the data, `x[short]`, has a shape that depends on the
values, which a trace cannot express. Compute for every element and mask the
result instead:

```python
# Kinetics.py, RxnKinetics positivity guard: before
short = conc + total_rates * limiter_dt < -eps * 10
if short.any():
    consumed = -total_rates[short] * limiter_dt
    scale = min(1.0, float(np.min(conc[short] / consumed)))
    rrates = rrates * max(scale, 0.0)

# after: same scale; exactly x1.0 when nothing is short
consumed = -total_rates * limiter_dt
ratio = np.where(short, conc / np.where(short, consumed, 1.0), np.inf)
rrates = rrates * np.clip(np.min(ratio), 0.0, 1.0)
```

The vessel's positivity limiter (`calculate_scale`) got the same treatment. It
keeps its numeric short cut (`return None` when nothing is throttled), and a
trace always computes the scales, where 1.0 is a no-op.

### 3.5 Scalar coercions and builtins (4 sites)

```python
self._mass = np.float64(value)          ->  self._mass = np.asarray(value, dtype=float)
max(requested_flow, 0.0)                ->  np.maximum(requested_flow, 0.0)
min(factor, scales[s.start])            ->  np.minimum(factor, scales[s.start])
float(np.min(...))                      ->  (keep the array)
```

### 3.6 Validation and type dispatch run on numbers only (1 site so far, more to come)

```python
if not np.isclose(value.sum(), 1):             # mass_frac setter
    raise ValueError("mass_frac must sum to one")
# ->
if not np.is_traced(value) and not np.isclose(value.sum(), 1):
    raise ValueError(...)
```

`isinstance(sup_sat, float)` in CrystKinetics is the same problem in another
form: a tracer is never a `float`, so traced code silently takes the array
branch.

### 3.7 State kept between calls (a rule for the backend, not a code edit)

- **Lazily initialised state.** `DefaultContinuousVesselVolume.observe` sets
  `target_volume` from the volume at the *first* call. Under tracing the first
  call is the trace, and it stored a tracer on the controller. Every backend
  already makes one numeric call before it integrates, and tracing must come
  after it.
- **Leaked tracers.** After a trace the vessel's phases hold traced values. A
  second trace through the Python objects (a separate `jit` for the Jacobian)
  reads them and fails. Rules: trace a *copy* of the vessel (`create_pseudo`
  already makes one), trace its Python code **once**, and derive everything else
  from the compiled function: `jit(jacfwd(rhs))`, not `jacfwd` of the method.
- **Controllers with memory** (a PI integral kept inside the controller) cannot
  be traced as they are. Inside an MPC they are replaced by inputs; for a
  closed-loop symbolic simulation the integral would have to become a state.
  That reopens the earlier decision that controllers own no solver states.

### 3.8 CasADi-only extras (probe stopped after five)

On top of 3.1-3.7, running the same code on `SX` needed:

| Problem | Example | Fix |
|---|---|---|
| Array **methods** SX lacks | `value.sum()`, `value.copy()` | `np.sum(value)`, `np.copy(value)` (~75 method calls on the RHS path) |
| **Attributes** SX lacks | `mass_frac.ndim == 1` (ThermoModule.py:387) | shape checks done before tracing, or in `xp` |
| **1-D semantics** SX cannot express | `np.dot(mass_frac, 1/rho)` | `xp.dot` treats every SX column as a numpy 1-D array |
| **Coercions** that reject symbols | `np.asarray(value, dtype=float)` | `xp.asarray` passes symbols through |
| **Silent NaN** | `float(sym)`, `buffer[i] = sym` | none in the framework; only tests catch it |

The silent NaN is the reason to develop against JAX even if CasADi is the
target: JAX stops at the exact line that would have produced it.

## 4. What does not change

| Part | Status |
|---|---|
| Existing backends (scipy, Assimulo, AssimuloDAE, diffeqpy) | untouched; they call the same code with numpy |
| Results and replay | numeric, as now; a traced backend returns numbers and `retrieve_results` runs unchanged |
| Mechanism / controller / vessel **interfaces** | unchanged: `get_solver_state_rates`, `actuate`, `unit_model` keep their signatures |
| Vessel assembly order, `StateCollection`, slices | unchanged |
| `ConcurrentVessel` | its iterated tears are not traceable, so a traced backend uses the **algebraic** tear mode, which maps directly onto IDA, CasADi's `idas`, or an implicit solve in JAX |
| Events | not traced. The root-finding restarts are a feature of the existing backends; a smooth model does not need them, and a CasADi/JAX backend can start without |

## 5. The new backends

```python
class JaxBackend(IntegratorBackend):          # sketch
    def compile_integrator(self, unit, **kw):
        unit.reset(); y0 = unit.create_solver_init_states()
        unit.unit_model(unit.elapsed_time, y0)            # numeric warm-up (3.7)
        pseudo = unit.create_pseudo()                     # trace a copy (3.7)
        self.rhs = jax.jit(lambda t, y: pseudo.unit_model(t, y))
        self.jac = jax.jit(jax.jacfwd(self.rhs, argnums=1))
        # integrate with diffrax, or hand rhs/jac to scipy/CVODE as exact Jacobians

class CasadiBackend(IntegratorBackend):       # sketch: the bridge in spike_bridge_casadi.py
    def compile_integrator(self, unit, **kw):
        ...same JAX compile...
        f = RHSCallback(self.rhs, self.jac)               # casadi.Callback, exact Jacobian
        self.F = ca.integrator('F', 'idas' if unit.has_algebraic_balance else 'cvodes',
                               {'x': x, 'ode': f(x)}, ...)
```

Either backend then serves an MPC or estimation layer as `f(x, u)`, once the
controllers' manipulated variables are exposed as inputs mapped onto
`OperatingKey`s.

## 6. How big is the rest

| Path | Status | Known sites |
|---|---|---|
| Continuous reactor, isothermal and energy balance | **done in the spike** | 32 edits: 105 added and 124 removed lines in 10 modules, plus the 63-line `xp.py` |
| Crystallizers (`OneDFVMMechanism`, moments, PBM, `CrystKinetics`) | first failure found: `moments[n] = ...` (Mechanisms.py:573) | about 14 array writes and about 114 `if`/`while`, of which a minority branch on values (upwind direction, growth/dissolution, zero-volume guards) |
| Filter / deliquoring (`SolidLiquidSep_Refactored`) | not probed | 2 writes, 13 `if` |
| `ConcurrentVessel` | not traced as is; use algebraic tears | coupling code only (`CoupledInlet.physical` uses `float`) |

So the reactor path is a few hundred changed lines. I'd expect the whole RHS
path to come to roughly two to three times that. Most of it is the mechanical
`set_at`/`add_at` and `where` rewrites, and the judgement is in the regime
switches, where both branches must stay finite.

## 7. How to keep it honest

1. **Numeric bit-identity.** Every step must leave the existing numeric results
   bit-identical. The spike did, on both reactor cases, and the regression
   dumps from the ConcurrentVessel work give a ready check.
2. **Traced equals numeric.** At random states, the compiled RHS matches
   `unit_model` to round-off.
3. **Exact against finite differences.** Compare away from the clamps. At
   exactly zero concentration the `maximum(conc, 0)` kinks make any derivative
   a one-sided choice: 18% apart at the reactor's start state, 8e-9 once every
   species is present.
4. **Fail loudly.** Run the suite under JAX, so a value Python needs is an error
   rather than a NaN.

## Files here

| File | What it is |
|---|---|
| `jax_reactor_spike.patch` | the JAX-traceable reactor: every edit, tagged `SPIKE [category]` |
| `casadi_probe.patch` | the time-boxed CasADi additions to `xp` and the two method fixes |
| `probe_trace.py` | the layer-by-layer probe of section 2 (runs on the repo as is) |
| `spike_driver.py` | traces, JITs and checks the reactor RHS against numpy |
| `spike_bridge_casadi.py` | CasADi CVODES integrating the JAX-compiled reactor |

The spike scripts expect the scratch copy at `%TEMP%\ppdoc\spike` with CasADi
and JAX in `%TEMP%\ppdoc\pylib`; neither is installed in `pharmNew`.
