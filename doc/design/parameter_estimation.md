# Parameters as compiled inputs, sensitivities, and parameter estimation

*Branch `ParamEst`, off `ControllerSolverStates` (2e7070b), 2026-10-27. Approved 2026-10-27 without the MPC phase: MPC will be its own addition later.*

## What exists today

| Piece | State |
|---|---|
| `compile_integrator` once, then loop | **Works.** Every backend keeps `_compiled`, so a second `solve_unit` on the same unit and backend skips the compile and continues from `unit.elapsed_time` (after `unit.reset()`, from the initial state). Measured on the CSTR: first solve 0.45 s, second 0.02 s on JaxAssimuloBackend, same compiled model. |
| Parameters on the traced backends | **Bug.** The compiled function closes over a copy of the unit, so every number it read is frozen. After `set_params` + `reset()`, JaxAssimuloBackend silently returns the old answer (C_C 0.11380; the new parameters give 0.11048; AssimuloBackend gets 0.11048). A controller attribute changed between MPC steps is frozen the same way. |
| `eval_sens` | Vestigial. AssimuloBackend sets `sensmethod` but never gives the problem `p0` or `rhs_sens`; its `rhs_sensitivity` / `jac_*_numerical` methods reference attributes that do not exist. Every other backend raises `NotImplementedError`. |
| Legacy estimation | `ParamEstim.ParameterEstimation(func, param_seed, x_data, y_data, measured_ind, optimize_flags, ...)`; `func(params, x_vals, **kwargs) -> (y, sens)` with `sens` stacked state by state (`reorder_sens`). LM in `LevMarq.py`, covariance, plots, `StatsModule.StatisticsClass`. Units supply `paramest_wrapper` (reset, `modify_phase`, `Kinetics.set_params`, `solve_unit(eval_sens=True)`); legacy sensitivities are Assimulo CVODES with `p0` and a hand-written `rhs_sens` (reactors; crystallizers only for moments). Only example: `doc/online_docs/examples/param_estimation_solved.ipynb` (Ziegler, legacy BatchReactor; data in `workshop_data/Case_study_1`). |

## Probes (throwaway, in the session scratchpad)

- **Kinetic parameters as inputs of the compiled function work with no change to
  Kinetics.py.** Inside the trace, `set_params(p)` is called on the copy's
  kinetics (found through `create_pseudo`'s deepcopy memo). On the CSTR, the
  FVM batch crystallizer and the moments continuous crystallizer, the compiled
  f(t, y, p) matches the numeric model to 1e-14 at p and at p moved 5 %, and the
  exact df/dp matches central differences to 1e-10 to 1e-11.
- **The replay's output states can be traced too.** `output_states_at(t, y)`
  compiled as h(t, y) matches the replay to 1e-16 on the CSTR and the FVM batch
  crystallizer, with finite dh/dy. This matters because measured quantities
  are mostly output states: the CSTR's solver state is `mass_j`, while
  `mole_conc` is an output.

## What it takes, by where it lives

### A. The fitting / MPC script (no library change)

- Build the unit, choose a backend, declare which parameters are inputs, solve once (compiles), then loop: assign parameters, `reset()` for a fresh experiment or not for an MPC step, `solve_unit`.
- MPC: a *model* unit for predictions alongside the *plant* unit; each step the model is put at the plant's state, a horizon is predicted with the moves as inputs (sensitivities give the gradient), the first move is applied to the plant for one step. The optimizer (scipy, IPOPT, ...) is the script's.
- Multi-experiment fits: one unit per experiment, each compiled once.

### B. Vessel and integrator changes

1. **`Parameters`** (new, `PharmaPy/Parameters.py`): named blocks, each an owner object plus how to read and write it. `Parameters.kinetics(unit)` finds every kinetics object in `unit.iter_mechanisms()` (`concat_params` / `set_params` / `name_params`); `add(name, owner, attribute)` declares anything else (an MPC move on a controller). `values()`, `assign(vector)`, `names`.
2. **`TracedModel(unit, time, states, parameters=None)`**: the compiled function becomes f(t, y, inputs, p); inside the trace each block is assigned on the copy's owner (through the memo). New `jac_params` (df/dp) and `sens_rhs(t, y, S, p)` = J_y S + J_p as one vmapped JVP (no dense J). `rhs`, `jac`, `jvp` read p from the declared owners at each solve, so assigning parameters and solving again is correct and does not recompile.
3. **`solve_unit(..., parameters=None, eval_sens=False)`** through every backend's `compile_integrator` / `solve`. Numeric backends: parameters are live already; nothing changes when `parameters` is not passed (regression dumps must stay bit-identical).
4. **Sensitivities**:
   - JaxAssimuloBackend: CVODES forward sensitivities with `p0`, the exact `rhs_sens` above, `pbar` from |p|, results from `p_sol` (Assimulo 3.7.3 supports `problem.rhs_sens`, checked).
   - AssimuloBackend: the legacy method without a hand-written derivative: `p0` plus an rhs that assigns p, CVODES difference quotients for df/dp. The no-JAX path and the reference for the exact one.
   - CasadiBackend: p as the CasADi DAE's `p`, the Callback's forward derivatives extended to p, CVODES forward sensitivities through CasADi.
   - Scipy / JaxBackend / Diffeqpy: refuse `eval_sens`, as now.
   - Results: `result.sensitivities[field]`, shape (n_times, dim, n_params) for every solver *and* output field (outputs as dh/dy S + dh/dp from the traced output map), and `result.param_names`. Sensitivities only for free parameters.
5. **Stale-model check (the frozen-number bug)**: at the start of every solve after the first, a traced backend evaluates the compiled right-hand side and the unit's own at the start state and at a nudged one. If they disagree, a number the trace froze has changed (kinetics, a controller setting, a feed, the utility): it warns and traces again at the current state, without resetting the unit. Declared parameters are read before the check, so changing them never triggers it.
6. **ConcurrentVessel**: `create_pseudo` takes the memo so member owners map to their copies; blocks are named per member.
7. ~~MPC building blocks~~: dropped at approval; MPC is a later, separate addition.

### C. `PharmaPy/ParamEstim_Refactored.py`

- `Experiment(unit, time_data, y_data, measured=('mole_conc_liquid0', ['B', 'C']), weights=None)`: measured is any result field (solver or output state) with columns by index or species name, or a callable.
- `ParameterEstimation(experiments, parameters=Parameters.kinetics, optimize_flags=None, ...)`: compiles each experiment's unit once, then for each iterate assigns theta to every experiment, `reset()`, solves with `eval_sens=True`, and builds residuals and their Jacobian from the sensitivities.
- Reuses the legacy machinery rather than copying it: it builds the legacy `func(params, x_vals, **kwargs) -> (y, sens)` per experiment in the legacy layout and drives `ParamEstim.ParameterEstimation`, so LM, covariance, `paramest_df`, the plots and `StatsModule` work unchanged.

## Phases

### Phase 1: parameters as compiled inputs (B1-B3, B5, B6)
Verified when: compiled f(t, y, p') equals the numeric model after `assign(p')` on the CSTR, FVM batch, moments continuous crystallizer and a ConcurrentVessel pair; a second solve after assigning new parameters equals a fresh AssimuloBackend run, with one trace; the stale-parameter case recompiles and warns; regression suite bit-identical.
Files: `PharmaPy/Parameters.py` (new), `TracedBackends.py`, `IntegratorBackends.py`, `MultiPhaseVessel.py`, `ConcurrentVessel.py`, `tests/test_parameters.py` (new).

- 1.1 `Parameters` + `TracedModel` inputs + `jac_params` (S). Blocked by: none.
- 1.2 `parameters=` through `solve_unit` and all backends; stale guard (S). Blocked by: 1.1.
- 1.3 ConcurrentVessel memo (XS). Blocked by: 1.1.

### Phase 2: sensitivities (B4)
Verified when: sensitivities match central differences of whole solves (tight tolerances) for the CSTR and the FVM batch crystallizer, for solver and output fields; JaxAssimulo exact and AssimuloBackend difference quotients agree; timing recorded for the FVM crystallizer with 2, 5 and 13 free parameters.
Files: `TracedBackends.py`, `IntegratorBackends.py`, `MultiPhaseVessel.py` (result fields), `tests/test_sensitivities.py` (new).

- 2.1 JaxAssimuloBackend sensitivities + output sensitivities (M). Blocked by: 1.2.
- 2.2 AssimuloBackend sensitivities; remove the vestigial stubs (S). Blocked by: 2.1.
- 2.3 CasadiBackend sensitivities (M). Blocked by: 2.1.

### Phase 3: ParamEstim_Refactored (C)
Verified when: synthetic data generated at theta* are recovered from a perturbed start (CSTR and FVM batch crystallizer); the Ziegler case (five temperatures, refactored BatchReactor) gives the same estimates as the legacy notebook's run within its confidence intervals; `StatisticsClass` runs on the result.
Files: `PharmaPy/ParamEstim_Refactored.py` (new), `tests/test_paramestim_refactored.py` (new).

- 3.1 `Experiment` + adapter to legacy `ParameterEstimation` (S). Blocked by: 2.1.
- 3.2 Ziegler comparison against legacy (S). Blocked by: 3.1.

### ~~Phase 4: MPC building blocks~~
Dropped at approval (2026-10-27).

## What was built (2026-10-27)

### Using it

```python
from PharmaPy.Parameters import Parameters
from PharmaPy.ParamEstim_Refactored import Experiment, ParameterEstimation

# A loop over parameter values: compile once, then assign, reset, solve.
P = Parameters.kinetics(unit)              # every kinetics object's concat_params
unit.solve_unit(time_grid=t, parameters=P)        # compiles (traced backends)
P.assign(theta); unit.reset()
unit.solve_unit(time_grid=t, parameters=P, eval_sens=True)
unit.result.sensitivities['mole_conc_liquid0']    # (time, species, free parameter)
unit.result.param_names

# Anything else as a parameter, e.g. a controller setting:
Parameters().add('target_volume', unit.controller, 'target_volume')

# Estimation: one unit per experiment.
experiments = [Experiment(reactor_at(T), t_data[T], c_data[T],
                          measured=('mole_conc_liquid0', ['B', 'C']), name=f'{T}K')
               for T in temperatures]
estimation = ParameterEstimation(experiments, optimize_flags=None)
params, covariance, info = estimation.estimate()
estimation.plot_data_model(); estimation.statistics(alpha=0.95).get_intervals()
```

### Pieces

| Piece | Where |
|---|---|
| `Parameters`, `ParameterBlock`, `find_kinetics` | `PharmaPy/Parameters.py` (new) |
| Parameters as inputs: `_Traced`, `TracedModel(unit, t, y, parameters)`, `refresh_parameters`, `jac_params`, `tangents`, `sens_rhs`, `matches` | `PharmaPy/TracedBackends.py` |
| Compiled replay for output sensitivities: `TracedOutputs` | `PharmaPy/TracedBackends.py` |
| Stale-model check and retrace: `_CompiledModelCheck` (JaxBackend, JaxAssimuloBackend, CasadiBackend) | `PharmaPy/TracedBackends.py` |
| `parameters=` on every backend's `compile_integrator`/`solve`; `needs_compile` (recompiles when the declaration changes) | `PharmaPy/IntegratorBackends.py` |
| AssimuloBackend sensitivities (CVODES difference quotients); vestigial stubs removed; sensitivities carried across events | `PharmaPy/IntegratorBackends.py` |
| JaxAssimuloBackend sensitivities (exact `rhs_sens`) | `PharmaPy/TracedBackends.py` |
| CasadiBackend sensitivities (free parameters as the DAE's `p`, `jac:xf:p`) | `PharmaPy/TracedBackends.py` |
| `retrieve_sensitivities`, `output_tangents_by_differences` | `PharmaPy/MultiPhaseVessel.py` |
| `create_pseudo(memo)`, members' solver-state sensitivities | `PharmaPy/ConcurrentVessel.py` |
| `RxnKinetics.set_params` traceable (`set_at`); `CrystKinetics.set_params` splits by the actual lengths (size-dependent growth put two growth values in dissolution); a missing comma in its 15 reformulated names | `PharmaPy/Kinetics.py` |
| `Experiment`, `ParameterEstimation` over legacy `ParamEstim`/`LevMarq`/`StatsModule` | `PharmaPy/ParamEstim_Refactored.py` (new) |

### Verification

- Regression dumps (scipy and assimulo, 172 arrays each, every refactored unit form and three flowsheets): **all 344 arrays bit-identical** to 2e7070b.
- `tests/test_parameters.py` (22), `tests/test_sensitivities.py` (12), `tests/test_paramestim_refactored.py` (5), plus the existing suite. With JAX and CasADi: 180 passed, 1 failed; without them: 132 passed, 48 skipped, 1 failed. The failure is the pre-existing one listed under Limits.
- The frozen-number bug, before and after: C_C after `set_params` + `reset()` on JaxAssimuloBackend was 0.11380 (old constants) and is now 0.11048 (= AssimuloBackend), with a warning; declared, the second solve takes 0.05-0.09 s with the same compiled model (one trace).
- Sensitivities against central differences of whole solves: CSTR 1e-7 (all three backends), FVM batch crystallizer < 5e-4 (limited by the differences). CasADi and JaxAssimulo agree to < 1e-6 on the CSTR; on the crystallizer they agree to about 1e-9 (e.g. dC/db = -2.96706963e-06 both).
- Ziegler, five temperatures, refactored BatchReactor vs legacy (zachary-dev), same seeds:

| | Legacy | AssimuloBackend | JaxAssimuloBackend |
|---|---|---|---|
| phi_1 | -2.0218 -3.3316 -2.1522 -2.3013 | -2.0218 -3.3314 -2.1523 -2.3013 | -2.0218 -3.3316 -2.1522 -2.3013 |
| phi_2 | 9.8070 9.6876 9.7695 9.6829 | 9.8070 9.6877 9.7695 9.6829 | 9.8070 9.6876 9.7695 9.6829 |
| largest difference / legacy std | | 0.001 | < 0.0005 |
| objective | 0.2154698 | 0.2154718 | 0.2154704 |
| evaluations, time | 39, 2.7 s | 29, 38 s | 32, 5.4 s (five compiles included) |

  The 95% interval table from `StatsModule` matches legacy's to the digits printed.

  Where JaxAssimulo's time goes (measured again without a profiler): compiling
  the five units 2.9 s (about 0.6 s each), then 2.2 s for the estimation. One
  solve with sensitivities costs 12.7 ms, against legacy's 14 ms (2.7 s over
  195 solves); 2.2 ms of it is the numeric replay of output states. So the
  gap to legacy is the compile, a fixed cost that does not grow with the
  number of evaluations. Each experiment compiles its own copy because the
  trace bakes in that unit's temperature. Legacy's reactor has hand-written
  Jacobians in numpy, which is already fast at 5 states, so JAX has no
  per-solve speed to win here. Plain AssimuloBackend is slow (38 s) because
  the refactored right-hand side is heavier Python than legacy's and CVODES
  difference-quotients the sensitivities through it.

### Timing: sensitivities on the FVM batch crystallizer

25 states, 1200 s, default tolerances (1e-6); first solve (includes compiling) / a second solve with new parameter values.

| Free parameters | JaxAssimulo | CasADi | AssimuloBackend (numeric) |
|---|---|---|---|
| none (no sensitivities) | 2.9 / 0.6 s | 1.5 / 0.2 s | 1.1 / 1.2 s |
| 2 | 5.3 / 1.6 s | 9.0 / 5.8 s | 7.2 / 7.3 s |
| 5 | 5.0 / 1.1 s | 22.8 / 16.4 s | |
| 13 | 5.0 / 2.6 s | | (124 s at 1e-7, earlier probe) |

JaxAssimulo is the backend to estimate with: the second solve with all 13 parameters costs about four times one without sensitivities. CasADi's forward sensitivities grow with the parameter count because they difference the derivative callback once more for their Newton matrix (`enable_fd`); exact second derivatives from JAX would remove that, if it is ever needed. The numeric AssimuloBackend is the no-JAX reference: fine on reactors (1 s on the CSTR), slow on the FVM crystallizer.

### Limits, as built

- A ConcurrentVessel gets each member's **solver-state** sensitivities only; its output states come from the joint replay, which is not differentiated.
- CasadiBackend refuses sensitivities with algebraic states (IDAS) or `locate_events=True`; ScipyBackend, JaxBackend, AssimuloDAEBackend and DiffeqpyBackend refuse them as before.
- The stale-model check compares the right-hand side at the start state and one nudged point. A frozen number that changes nothing there (a set-point change scheduled for later in the run, held as a plain attribute) would not be caught; declaring it is the remedy.
- Sensitivities are with respect to declared parameters, not initial conditions.
- `Experiment` takes one time vector per experiment (legacy's per-state time vectors with missing data are not wired through).
- `tests/test_solid_liquid_sep_refactored.py::test_deliquoring_profile_matches_the_original[0.01]` fails (1.25e-4 against rtol 1e-4) on 2e7070b too; not touched here.

Docs at the end: this file becomes the design record (results, timings), as `controller_solver_states.md` did.

## Decision log

| Decision | Chosen option | Why |
|---|---|---|
| How parameters reach the trace | `set_params` / setattr on the copy's owner, inside the trace | Probed: works on all unit forms with no change to Kinetics.py or mechanisms. |
| Sensitivity RHS | One vmapped JVP over (y, p) | Exact, no dense Jacobian, works under SPGMR for large units. |
| Output sensitivities | Trace `output_states_at` | Probed: exact; measured quantities are mostly output states. |
| Estimation optimizer, statistics, plots | Reuse legacy ParamEstim / LevMarq / StatsModule through an adapter | Fidelity first; no second copy of ~870 lines. Approved. |
| Experiments | One unit per experiment, each compiled once | Approved. Legacy changed one unit per experiment, which would need every per-experiment setting declared as an input. |
| Sensitivity backends | JaxAssimulo (exact), then AssimuloBackend (difference quotients), then CasADi | Approved. |
| MPC | Not in this branch | The user will add it separately. |
| Frozen numbers on traced backends | Compare compiled and numeric right-hand sides at the start of each solve; on a mismatch warn and trace again, keeping the unit's state | Changed from the planned kinetics fingerprint: the user asked for the bug to be fixed, and a fingerprint of the kinetics would still miss a controller setting, a feed or the utility. The comparison catches any frozen number that changes the right-hand side near the start state. Approved outcome (recompile and warn) unchanged. |
| Sensitivities across events | Carry them over in the event handler (`solver.yS0`) | Probed: Assimulo restarts sensitivities from `yS0` (zeros) at every event, giving dy/dp = -0.297 instead of -0.410 after one event in a test problem; with the carry, -0.410425 (exact). PharmaPy's events mark corners where the right-hand side is continuous, so the sensitivities do not jump there. |

## Not yet specified

- Initial-condition sensitivities (estimating initial concentrations, multiple-shooting MPC): not in this plan.
- Flowsheet-level estimation (`SimulationExec.SetParamEstimation`) for refactored units: not in this plan.
- Phase 2.3: whether CasADi's forward sensitivities through a Callback are fast enough on a 205-state crystallizer: fog, measured at pickup.

## Always / Never

- Always: no change in results when `parameters` is not passed (regression dumps bit-identical); verify on reactor and crystallizer forms.
- Never: an MPC optimizer or controller class in the library; changes to legacy ParamEstim.py beyond what the adapter needs; commits without the user asking.

## Progress

| Phase | Status |
|---|---|
| 1 Parameters as inputs | done |
| 2 Sensitivities | done (2.1 JaxAssimulo, 2.2 Assimulo, 2.3 CasADi) |
| 3 ParamEstim_Refactored | done (Ziegler matches legacy) |
| 4 MPC blocks | dropped |
