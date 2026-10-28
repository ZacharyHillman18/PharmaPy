"""Parameter sensitivities: solve_unit(eval_sens=True).

What is pinned here:

- result.sensitivities[field] is d(field)/d(free parameter) for solver
  states and output states alike, and agrees with central differences of
  whole solves: AssimuloBackend (CVODES difference quotients),
  JaxAssimuloBackend (exact sensitivity right-hand side) and CasadiBackend
  (CasADi's forward sensitivities), on the CSTR and the FVM batch
  crystallizer.
- Sensitivities survive an event: CVode restarts them from zero at every
  event unless they are carried over.
- New parameter values between sensitivity solves need no new trace.
- A ConcurrentVessel gets each member's solver-state sensitivities.
- Backends without sensitivities refuse eval_sens.

Units: time [s], mass [kg], mole_conc [kmol/m**3].
"""

import numpy as np
import pytest

from PharmaPy.IntegratorBackends import (AssimuloBackend, ScipyBackend,
                                         attach_assimulo_events)
from PharmaPy.Parameters import Parameters

from test_traced_backends import _crystallizer, _max_rel, _pair, _reactor

pytestmark = pytest.mark.unit

REACTOR_GRID = np.linspace(0.0, 3600.0, 13)  # [s]
CRYST_GRID = np.linspace(0.0, 1200.0, 9)  # [s]
REFERENCE = {'atol': 1e-11, 'rtol': 1e-11}

# s_1, k_g, E_g, g: orders, a rate constant and a zero activation energy.
# (Primary nucleation is left out: its effect is small enough that
# differences of whole solves are only good to 1e-4 even at 1e-11.)
CRYST_FREE = [5, 7, 8, 9]
CRYST_REFERENCE = {'atol': 1e-10, 'rtol': 1e-10}


def _step(value):
    """A central-difference step. A zero activation energy gets 1 J/mol:
    1e-5 of nothing would move the rates by less than the solver's noise."""
    return 1e-5 * abs(value) if value != 0 else 1.0


def _differences(build, grid, free, fields, reference=REFERENCE):
    """d(field)/d(parameter) by central differences of whole solves."""
    unit = build(AssimuloBackend(options=dict(reference)))
    values = Parameters.kinetics(unit).values()

    columns = {field: [] for field in fields}

    for index in free:
        step = _step(values[index])
        ends = []

        for sign in (1, -1):
            unit = build(AssimuloBackend(options=dict(reference)))
            moved = values.copy()
            moved[index] += sign * step
            Parameters.kinetics(unit).assign(moved)
            unit.solve_unit(time_grid=grid, verbose=False)
            ends.append(unit.result)

        for field in fields:
            high, low = (np.asarray(getattr(end, field), dtype=float)
                         for end in ends)
            columns[field].append((high - low) / (2 * step))

    return {field: np.stack(columns[field], axis=-1) for field in fields}


def _assert_sensitivities_match(unit, grid, expected, tolerance):
    rows = [int(np.flatnonzero(np.isclose(unit.result.time, t))[-1])
            for t in grid]

    for field, reference in expected.items():
        computed = unit.result.sensitivities[field][rows]

        for column in range(reference.shape[-1]):
            assert _max_rel(computed[..., column],
                            reference[..., column]) < tolerance, (field,
                                                                  column)


def _sensitivity_solve(build, integrator, grid, free=None):
    unit = build(integrator)
    parameters = Parameters.kinetics(unit)

    if free is not None:
        mask = np.zeros(parameters.size, dtype=bool)
        mask[free] = True
        parameters.free = mask

    unit.solve_unit(time_grid=grid, verbose=False, parameters=parameters,
                    eval_sens=True)

    return unit


REACTOR_FIELDS = ('mass_j_liquid0', 'mole_conc_liquid0')
CRYST_FIELDS = ('mass_j_liquid0', 'distrib_solid0', 'supersat', 'mu_n')


@pytest.fixture(scope="module")
def reactor_differences():
    pytest.importorskip("assimulo")
    return _differences(lambda integrator: _reactor(integrator=integrator),
                        REACTOR_GRID, range(4), REACTOR_FIELDS)


@pytest.fixture(scope="module")
def crystallizer_differences():
    pytest.importorskip("assimulo")
    return _differences(
        lambda integrator: _crystallizer("batch", integrator=integrator),
        CRYST_GRID, CRYST_FREE, CRYST_FIELDS, CRYST_REFERENCE)


# ---------------------------------------------------------------------
# No JAX needed
# ---------------------------------------------------------------------

class _Threshold:
    """The event surface y - 0.5 of dy/dt = -a y, as a vessel shows it."""

    def evaluate_events(self, time, states):
        return np.array([states[0] - 0.5])

    def handle_event(self, time, indices):
        return False


@pytest.mark.parametrize("carry", [False, True])
def test_sensitivities_are_carried_across_an_event(carry):
    """dy/da of dy/dt = -a y at t = 5 is -5 exp(-5 a). CVode restarts at
    the event at y = 0.5 and, uncarried, from yS0 = 0: -0.297, not -0.410."""
    assimulo = pytest.importorskip("assimulo")
    from assimulo.problem import Explicit_Problem
    from assimulo.solvers import CVode

    a = 0.5

    problem = Explicit_Problem(lambda t, y, p=None, sw=None: -p * y,
                               np.array([1.0]), t0=0.0, p0=np.array([a]),
                               sw0=[True])
    problem.rhs_sens = lambda t, y, s, p, sw=None: -p[0] * s - y[:, None]
    attach_assimulo_events(problem, _Threshold(), carry_sensitivities=carry)

    solver = CVode(problem)
    solver.sensmethod = "SIMULTANEOUS"
    solver.report_continuously = True
    solver.verbosity = 50
    solver.atol = solver.rtol = 1e-10
    solver.simulate(5.0)

    exact = -5.0 * np.exp(-5.0 * a)
    final = np.asarray(solver.p_sol)[0, -1, 0]

    if carry:
        assert final == pytest.approx(exact, rel=1e-6)
    else:
        assert final != pytest.approx(exact, rel=1e-2)


def test_assimulo_backend_sensitivities_match_differences(
        reactor_differences):
    unit = _sensitivity_solve(lambda integrator: _reactor(
        integrator=integrator), AssimuloBackend(
            options={'atol': 1e-8, 'rtol': 1e-8}), REACTOR_GRID)

    assert unit.result.param_names == ['k_1', 'k_2', 'E_{a, 1}', 'E_{a, 2}']
    assert unit.result.sensitivities['mole_conc_liquid0'].shape == (
        REACTOR_GRID.size, 5, 4)

    _assert_sensitivities_match(unit, REACTOR_GRID, reactor_differences,
                                1e-4)


def test_the_unit_keeps_its_nominal_parameters():
    """CVODES's difference quotients evaluate the model at perturbed
    parameters; the unit has to be left at the nominal ones."""
    pytest.importorskip("assimulo")

    unit = _reactor(integrator=AssimuloBackend(options={}))
    parameters = Parameters.kinetics(unit)
    before = parameters.values()

    unit.solve_unit(time_grid=REACTOR_GRID, verbose=False,
                    parameters=parameters, eval_sens=True)

    assert np.array_equal(parameters.values(), before)


def test_backends_without_sensitivities_refuse():
    unit = _reactor(integrator=ScipyBackend())

    with pytest.raises(NotImplementedError):
        unit.solve_unit(time_grid=REACTOR_GRID, verbose=False,
                        eval_sens=True)


# ---------------------------------------------------------------------
# JAX
# ---------------------------------------------------------------------

@pytest.fixture(scope="module")
def jax():
    return pytest.importorskip("jax")


def _sensitivity_backend(name, tolerance):
    from PharmaPy.TracedBackends import CasadiBackend, JaxAssimuloBackend

    options = {'atol': tolerance, 'rtol': tolerance}

    if name == "casadi":
        pytest.importorskip("casadi")
        return CasadiBackend(options=options)

    return JaxAssimuloBackend(options=options)


@pytest.mark.parametrize("backend", ["jax_cvode", "casadi"])
def test_reactor_sensitivities_match_differences(jax, backend,
                                                 reactor_differences):
    unit = _sensitivity_solve(lambda integrator: _reactor(
        integrator=integrator), _sensitivity_backend(backend, 1e-10),
        REACTOR_GRID)

    _assert_sensitivities_match(unit, REACTOR_GRID, reactor_differences,
                                1e-5)


def test_crystallizer_sensitivities_match_differences(
        jax, crystallizer_differences):
    """Solver states (liquid masses, the distribution) and output states
    (supersaturation, moments) alike. JaxAssimuloBackend only: CasADi's
    forward sensitivities take some ten times as long here (they difference
    the derivative callback once more), and the reactor tests cover them."""
    unit = _sensitivity_solve(
        lambda integrator: _crystallizer("batch", integrator=integrator),
        _sensitivity_backend("jax_cvode", 1e-9), CRYST_GRID,
        free=CRYST_FREE)

    assert unit.result.param_names == ['s_1', 'k_{g}', 'E_{g}', 'g']

    # The differences, not the sensitivities, set this: solves at 1e-10
    # make them good to a few 1e-4 on a nucleating population.
    _assert_sensitivities_match(unit, CRYST_GRID, crystallizer_differences,
                                5e-4)


def test_casadi_and_cvode_sensitivities_agree(jax):
    """Two integrators that share nothing but the compiled model: CVODES
    under Assimulo with the exact sensitivity right-hand side, and CasADi's
    own forward sensitivities."""
    def build(integrator):
        return _reactor(integrator=integrator)

    cvode = _sensitivity_solve(build, _sensitivity_backend("jax_cvode",
                                                           1e-10),
                               REACTOR_GRID)
    casadi = _sensitivity_solve(build, _sensitivity_backend("casadi", 1e-10),
                                REACTOR_GRID)

    for field in REACTOR_FIELDS + ('vessel_vol', 'Total_m_in_vessel'):
        assert _max_rel(casadi.result.sensitivities[field],
                        cvode.result.sensitivities[field]) < 1e-6, field


def test_new_parameters_between_sensitivity_solves_need_no_new_trace(jax):
    from PharmaPy.TracedBackends import JaxAssimuloBackend

    def build(integrator):
        return _reactor(integrator=integrator)

    unit = _sensitivity_solve(build, JaxAssimuloBackend(options={}),
                              REACTOR_GRID)
    model = unit.integrator.model

    parameters = unit.integrator.parameters
    parameters.assign(parameters.values() * 1.2)
    unit.reset()
    unit.solve_unit(time_grid=REACTOR_GRID, verbose=False,
                    parameters=parameters, eval_sens=True)

    fresh = build(JaxAssimuloBackend(options={}))
    fresh_parameters = Parameters.kinetics(fresh)
    fresh_parameters.assign(parameters.values())
    fresh.solve_unit(time_grid=REACTOR_GRID, verbose=False,
                     parameters=fresh_parameters, eval_sens=True)

    assert unit.integrator.model is model and model.traces == 1

    for field in REACTOR_FIELDS:
        assert _max_rel(unit.result.sensitivities[field],
                        fresh.result.sensitivities[field]) < 1e-8, field


def test_a_concurrent_vessel_gives_each_member_its_sensitivities(jax):
    from PharmaPy.TracedBackends import JaxAssimuloBackend

    def build(integrator):
        return _pair(integrator)

    grid = np.linspace(0.0, 3600.0, 7)
    free = [0, 1]  # R01's k_1, k_2

    unit = build(JaxAssimuloBackend(options={'atol': 1e-10,
                                             'rtol': 1e-10}))
    parameters = Parameters.kinetics(unit)
    mask = np.zeros(parameters.size, dtype=bool)
    mask[free] = True
    parameters.free = mask

    unit.solve_unit(time_grid=grid, verbose=False, parameters=parameters,
                    eval_sens=True)

    values = parameters.values()

    for column, index in enumerate(free):
        step = 1e-5 * abs(values[index])
        ends = []

        for sign in (1, -1):
            other = build(AssimuloBackend(options=dict(REFERENCE)))
            moved = values.copy()
            moved[index] += sign * step
            Parameters.kinetics(other).assign(moved)
            other.solve_unit(time_grid=grid, verbose=False)
            ends.append(other)

        for name in ('R01', 'R02'):
            expected = (np.asarray(ends[0][name].result.mass_j_liquid0)
                        - np.asarray(ends[1][name].result.mass_j_liquid0)
                        ) / (2 * step)
            computed = unit[name].result.sensitivities['mass_j_liquid0']

            assert _max_rel(computed[..., column], expected) < 1e-5, name


def test_jax_backend_refuses_sensitivities(jax):
    from PharmaPy.TracedBackends import JaxBackend

    unit = _reactor(integrator=JaxBackend())

    with pytest.raises(NotImplementedError):
        unit.solve_unit(time_grid=REACTOR_GRID, verbose=False,
                        eval_sens=True)
