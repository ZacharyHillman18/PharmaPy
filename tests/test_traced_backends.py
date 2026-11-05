"""The traced right-hand side, and the backends that compile it.

PharmaPy.xp lets a vessel's own unit_model run under JAX's tracer;
PharmaPy.TracedBackends compiles it once and gives integrators its exact
derivatives (JaxBackend on scipy, JaxAssimuloBackend on CVode, CasadiBackend
on CasADi's CVODES).

What is pinned here:

- xp leaves numeric callers alone: numpy in, numpy out, writes in place.
- The compiled right-hand side equals the unit's own, for each unit form,
  at the start and away from it, before and after a temperature profile
  written as plain Python switches branch.
- The exact Jacobian agrees with central differences away from kinks.
- Each backend reproduces AssimuloBackend.
- A population balance whose net mass rate is exactly zero is counted once
  (this one needs no JAX).

Everything that needs JAX skips without it, and the CasADi tests skip without
CasADi.

Units: vol [m**3], vol_flow [m**3/s], mass_conc [kg/m**3], temp [K],
time [s].
"""

import os

import numpy as np
import pytest

from PharmaPy import xp
from PharmaPy.Crystallizers_Refactored import (BatchCrystallizer,
                                               ContinuousCrystallizer,
                                               SemiBatchCrystallizer)
from PharmaPy.IntegratorBackends import AssimuloBackend
from PharmaPy.Kinetics import CrystKinetics, RxnKinetics
from PharmaPy.Mechanisms import MomentsPopulationBalance, OneDFVMMechanism
from PharmaPy.Phases_Refactored import LiquidPhase, SolidPhase
from PharmaPy.ProcessControl_Refactored import (ContinuousVesselController,
                                                SimpleTemperatureController)
from PharmaPy.Reactors_Refactored import ContinuousReactor
from PharmaPy.Streams_Refactored import LiquidStream
from PharmaPy.Utilities import CoolingWater

pytestmark = pytest.mark.unit

DATA_PATH = os.path.join(
    os.path.dirname(__file__), "Flowsheet", "data", "compound_database.json"
)

H_CONV = 10000.0  # [W/(m**2 K)]
DIAMETER = 0.4  # [m]
VOLUME = 0.06  # [m**3]
FEED_VOLFLOW = 1e-5  # [m**3/s]

# The continuous reactor of FlowsheetTester and ConcurrentTester.
RXNS = ['A + B --> C', 'C + A --> D']
K_VALS = np.array([2.654e4, 5.3e2])
EA_VALS = np.array([4.0e4, 3.0e4])
TEMP_RXN = 313.15  # [K]
CONC_RXN = np.array([0.33, 0.33, 0, 0, 0])  # [kmol/m**3]
FEED_RXN = np.array([0.5, 0.5, 0, 0, 0])  # [kmol/m**3]

# The crystallization chemistry of FlowsheetTester, nucleation per m3 slurry.
SOLUB = np.array([2.269e2, -1.88e0, 3.89e-3])
PRIM = (3e8 * VOLUME, 0, 3)
SEC = (4.46e10 * VOLUME, 0, 2, 1e-5)
GROWTH = (5, 0, 1.32)
DISSOL = (1, 0, 1)
CONC_C = np.array([0., 0., 40., 0., 0.])  # [kg/m**3], supersaturated
TEMP_START = 298.15  # [K]
TEMP_END = 278.15  # [K]
X_GRID = np.geomspace(1, 1500, num=20)  # [micron]

COOLING_TIME = 1800.0  # [s]


def cooling(time):
    """Linear cooling, then a hold, as plain Python.

    A trace cannot look inside the ``if``; the compiled right-hand side runs
    this on the number through xp.of_time.
    """
    if time < COOLING_TIME:
        return TEMP_START + (TEMP_END - TEMP_START) * time / COOLING_TIME

    return TEMP_END


def _reactor(integrator=None, controller=None, fed=True):
    """The continuous reactor; one fed through a link has no feed of its own."""
    if controller is None:
        controller = ContinuousVesselController(temp_func=lambda t: TEMP_RXN)

    unit = ContinuousReactor(integrator=integrator, h_conv=H_CONV,
                             diam=DIAMETER, controller=controller)
    unit.Phases = LiquidPhase(DATA_PATH, temp=TEMP_RXN, mole_conc=CONC_RXN,
                              vol=VOLUME, name_solv='solvent')
    unit.RxnKinetics = RxnKinetics(path=DATA_PATH, rxn_list=RXNS,
                                   k_params=K_VALS, ea_params=EA_VALS)

    if fed:
        unit.Inlet = LiquidStream(DATA_PATH, temp=TEMP_RXN,
                                  mole_conc=FEED_RXN, vol_flow=FEED_VOLFLOW,
                                  name_solv='solvent')

    return unit


def _solid(form):
    solid = SolidPhase(DATA_PATH, mass=0.0, mass_frac=[0, 0, 1, 0, 0])

    if form == "fvm":
        solid.mechanisms = OneDFVMMechanism(
            solid, target_components='C', solvent_name='solvent',
            x_grid=X_GRID, distrib_init=np.zeros_like(X_GRID))
    else:
        solid.mechanisms = MomentsPopulationBalance(
            owning_phase=solid, target_components='C',
            solvent_name='solvent', moments_init=np.zeros(4))

    return solid


def _crystallizer(mode, form="fvm", integrator=None):
    """A cooling crystallizer, batch, semibatch or continuous."""

    if mode == "continuous":
        unit = ContinuousCrystallizer(
            integrator=integrator, h_conv=H_CONV, diam=DIAMETER,
            controller=ContinuousVesselController(temp_func=cooling))
    else:
        cls = BatchCrystallizer if mode == "batch" else SemiBatchCrystallizer
        unit = cls(integrator=integrator, h_conv=H_CONV, diam=DIAMETER,
                   controller=SimpleTemperatureController(temp_func=cooling))

    unit.Phases = [LiquidPhase(DATA_PATH, temp=TEMP_START, mass_conc=CONC_C,
                               vol=VOLUME, name_solv='solvent'),
                   _solid(form)]
    unit.CrystKinetics = CrystKinetics(SOLUB, nucl_prim=PRIM, nucl_sec=SEC,
                                       growth=GROWTH, dissolution=DISSOL)

    if mode != "batch":
        unit.Inlet = LiquidStream(DATA_PATH, temp=TEMP_START,
                                  mass_conc=CONC_C, vol_flow=FEED_VOLFLOW,
                                  name_solv='solvent')

    unit.Utility = CoolingWater(mass_flow=1, temp_in=TEMP_END)

    return unit


UNITS = {
    "reactor": _reactor,
    "batch_fvm": lambda: _crystallizer("batch"),
    "semibatch_fvm": lambda: _crystallizer("semibatch"),
    "continuous_fvm": lambda: _crystallizer("continuous"),
    "continuous_moments": lambda: _crystallizer("continuous", "moments"),
}


def _start(unit):
    """The unit laid out as its backend would see it before integrating."""
    unit.compile_structure()
    unit.reset()
    states = np.asarray(unit.create_solver_init_states(), dtype=float)
    return float(unit.elapsed_time), states


def _numeric_rhs(unit, time, states):
    return np.array(unit.unit_model(time, states), dtype=float)


def _max_rel(a, b):
    return np.max(np.abs(a - b)) / np.max(np.abs(b))


# ---------------------------------------------------------------------
# No JAX needed
# ---------------------------------------------------------------------

def test_xp_leaves_numeric_callers_alone():
    values = np.array([0.0, 1.0, -2.0])

    assert type(xp.exp(values)) is np.ndarray
    assert type(xp.where(values > 0, values, 0.0)) is np.ndarray
    assert not xp.is_traced(values)

    # In place, on the same object, exactly as the plain assignment.
    out = xp.set_at(values, 1, 5.0)
    assert out is values and values[1] == 5.0

    out = xp.add_at(values, slice(0, 2), 1.0)
    assert out is values and list(values[:2]) == [1.0, 6.0]

    assert xp.of_time(cooling, 900.0) == cooling(900.0)
    assert type(xp.as_float(np.float64(2.5))) is float


def test_a_zero_net_rate_counts_the_population_balance_once():
    """A moments crystallizer starts with mu_2 = 0, so its net mass rate is
    exactly zero while it nucleates. Both directions of the reversible
    transfer used to keep the result there, and mu_0 grew at twice the rate
    it has a moment later.
    """
    unit = _crystallizer("batch", "moments")
    time, states = _start(unit)

    mu = unit.solver_state_collection.slices[
        next(key for key in unit.solver_state_collection.keys
             if key.name == "mu_n")]

    at_zero = _numeric_rhs(unit, time, states)[mu][0]

    nudged = states.copy()
    nudged[mu.start + 2] = 1e-12  # [micron**2/m**3], barely any surface
    beside = _numeric_rhs(unit, time, nudged)[mu][0]

    assert at_zero > 0
    assert at_zero == pytest.approx(beside, rel=1e-9)


# ---------------------------------------------------------------------
# JAX
# ---------------------------------------------------------------------

@pytest.fixture(scope="module")
def jax():
    return pytest.importorskip("jax")


@pytest.mark.parametrize("name", sorted(UNITS))
def test_compiled_rhs_matches_the_unit(jax, name):
    from PharmaPy.TracedBackends import TracedModel

    unit = UNITS[name]()
    time, states = _start(unit)
    model = TracedModel(unit, time, states)

    rng = np.random.default_rng(0)
    elsewhere = (states * (1 + 1e-2 * rng.standard_normal(states.size))
                 + 1e-6 * (states == 0))

    # Both sides of the profile's switch at COOLING_TIME.
    for when, at in ((time, states), (time + 900.0, elsewhere),
                     (time + 2500.0, elsewhere)):
        numeric = _numeric_rhs(unit, when, at)
        assert _max_rel(model.rhs(when, at), numeric) < 1e-12, when

    assert model.traces == 1


def test_exact_jacobian_matches_central_differences(jax):
    from PharmaPy.TracedBackends import TracedModel

    unit = _reactor()
    time, states = _start(unit)

    # Every species present, away from the clamps at zero concentration,
    # where a derivative is only a one-sided choice.
    states = states + 0.05 * np.max(states)
    model = TracedModel(unit, time, states)

    exact = model.jac(time, states)
    differences = np.empty_like(exact)

    for column in range(states.size):
        step = 1e-6 * max(abs(states[column]), 1e-3)
        up, down = states.copy(), states.copy()
        up[column] += step
        down[column] -= step
        differences[:, column] = (_numeric_rhs(unit, time, up)
                                  - _numeric_rhs(unit, time, down)) / (2 * step)

    assert _max_rel(exact, differences) < 1e-6
    assert model.patched_columns == 0


def test_jax_assimulo_options_follow_assimulo_backend(jax):
    """An empty dict means no options, as for AssimuloBackend; only None
    takes the default. (Empty used to fall to maxh=1: 21,600 steps on a
    six-hour CSTR.) And no two instances share one dict."""
    from PharmaPy.TracedBackends import JaxAssimuloBackend

    assert JaxAssimuloBackend(options={}).options == {}
    assert JaxAssimuloBackend().options == {'maxh': 1}

    first, second = JaxAssimuloBackend(), JaxAssimuloBackend()
    first.options['maxh'] = 5
    assert second.options == {'maxh': 1}


def test_a_second_trace_is_refused(jax):
    """Tracing again would run the copy's objects on what the first trace
    left behind them; a different argument type would cause it."""
    from PharmaPy.TracedBackends import TracedModel

    unit = _reactor()
    time, states = _start(unit)
    model = TracedModel(unit, time, states)

    with pytest.raises(RuntimeError, match="traced a second time"):
        model._rhs(np.float32(time), states.astype(np.float32),
                   model.inputs(time), model.parameter_values)


def _final(unit):
    result = unit.result
    names = ("mass_j_liquid0", "distrib_solid0", "temp", "vol")
    return {name: np.asarray(getattr(result, name))[-1]
            for name in names if hasattr(result, name)}


def _backend(name):
    from PharmaPy.TracedBackends import (CasadiBackend, JaxAssimuloBackend,
                                         JaxBackend)

    if name == "casadi":
        pytest.importorskip("casadi")
        return CasadiBackend(options={'maxh': 60})

    if name == "jax_cvode":
        return JaxAssimuloBackend(options={'maxh': 60})

    return JaxBackend(options={'maxh': 60})


@pytest.fixture(scope="module")
def reference():
    """AssimuloBackend's answers, the ones the traced backends must match."""
    pytest.importorskip("assimulo")

    answers = {}

    for name, build, runtime in (("reactor", _reactor, 3600.0),
                                 ("crystallizer",
                                  lambda integrator: _crystallizer(
                                      "batch", integrator=integrator),
                                  1200.0)):
        unit = build(integrator=AssimuloBackend(options={'maxh': 60}))
        unit.solve_unit(runtime=runtime, verbose=False)
        answers[name] = (runtime, _final(unit))

    return answers


@pytest.mark.parametrize("backend", ["jax", "jax_cvode", "casadi"])
@pytest.mark.parametrize("name", ["reactor", "crystallizer"])
def test_backend_reproduces_assimulo(jax, reference, backend, name):
    runtime, expected = reference[name]

    if name == "reactor":
        unit = _reactor(integrator=_backend(backend))
    else:
        unit = _crystallizer("batch", integrator=_backend(backend))

    unit.solve_unit(runtime=runtime, verbose=False)
    final = _final(unit)

    for key, value in expected.items():
        assert _max_rel(final[key], value) < 1e-4, key

    assert unit.integrator.model.traces == 1


# ---------------------------------------------------------------------
# ConcurrentVessel
# ---------------------------------------------------------------------

C_INDEX = 2  # species C
CV_GRID = np.linspace(0.0, 3 * 3600.0, 31)  # [s]


class FeedFromDownstream(ContinuousVesselController):
    """R01's feed set from R02's [C], written to be traced.

    The same controller as in test_concurrent_vessel.py, with its two
    numeric habits replaced: the measurement stays an array element rather
    than a float, and the floor on the feed is np.maximum rather than max.
    Under a trace both would have needed the value.
    """

    def __init__(self, downstream, c_set, gain=8.0):
        super().__init__(temp_func=lambda t: TEMP_RXN)

        self.downstream = downstream
        self.c_set = c_set
        self.gain = gain

    def actuate(self, time, completed_state, unit, resolved_inlets=None,
                resolved_outlets=None):

        if resolved_inlets is None:
            from PharmaPy.DataClasses import OperatingKey, PhaseRef

            measured = self.downstream.Phases.Liquids[0].mole_conc[C_INDEX]
            error = (self.c_set - measured) / self.c_set

            key = OperatingKey('vol_flow', connection=0,
                               phaseref=PhaseRef('liquid', 0), port='inlet')
            self.operating_conditions[key] = (
                FEED_VOLFLOW * xp.maximum(1.0 + self.gain * error, 0.05))
            return

        super().actuate(time, completed_state, unit, resolved_inlets,
                        resolved_outlets)


def _pair(integrator, recycle=False, controlled=False):
    from PharmaPy.ConcurrentVessel import ConcurrentVessel, StreamLink

    r02 = _reactor(fed=False)
    r01 = _reactor(controller=FeedFromDownstream(r02, c_set=0.13)
                   if controlled else None)

    links = [StreamLink('R01', 'R02')]

    if recycle:
        links.append(StreamLink('R02', 'R01', fraction=0.5))

    return ConcurrentVessel({'R01': r01, 'R02': r02}, links,
                            integrator=integrator,
                            tear_mode='algebraic' if recycle else 'iterate')


def _pair_final(cv):
    return np.concatenate([np.asarray(cv[name].result.mass_j_liquid0)[-1]
                           for name in ('R01', 'R02')])


@pytest.mark.parametrize("controlled", [False, True])
def test_concurrent_vessel_on_jax_and_casadi(jax, controlled):
    """An acyclic pair, and the same pair with R01's feed set from R02."""
    pytest.importorskip("casadi")
    from PharmaPy.IntegratorBackends import ScipyBackend

    expected = _pair(ScipyBackend(options={'maxh': 60}),
                     controlled=controlled)
    expected.solve_unit(time_grid=CV_GRID, verbose=False)

    for backend in ("jax", "casadi"):
        cv = _pair(_backend(backend), controlled=controlled)
        cv.solve_unit(time_grid=CV_GRID, verbose=False)

        assert _max_rel(_pair_final(cv), _pair_final(expected)) < 1e-5


def test_recycle_as_algebraic_tears_on_casadi_idas(jax):
    pytest.importorskip("casadi")
    pytest.importorskip("assimulo")
    from PharmaPy.IntegratorBackends import AssimuloDAEBackend

    expected = _pair(AssimuloDAEBackend(options={'maxh': 60}), recycle=True)
    expected.solve_unit(time_grid=CV_GRID, verbose=False)

    cv = _pair(_backend("casadi"), recycle=True)
    cv.solve_unit(time_grid=CV_GRID, verbose=False)

    assert cv.has_algebraic_balance
    assert _max_rel(_pair_final(cv), _pair_final(expected)) < 1e-5


def test_a_crystallizer_train_traces_through_its_slurry_link(jax):
    """A continuous crystallizer feeding another sends a slurry: liquid and
    crystals. Splitting that stream's flow between its phases used float()
    and the builtin max/min, which a trace cannot take, so no crystallizer
    train compiled. The compiled right-hand side now equals the train's own,
    with crystals in the stream (a nonzero solid share)."""
    from PharmaPy.ConcurrentVessel import ConcurrentVessel, StreamLink
    from PharmaPy.TracedBackends import TracedModel

    cv = ConcurrentVessel({'CR01': _crystallizer("continuous"),
                           'CR02': _crystallizer("continuous")},
                          [StreamLink('CR01', 'CR02')])
    time, states = _start(cv)

    rng = np.random.default_rng(2)
    elsewhere = (states * (1 + 1e-2 * rng.standard_normal(states.size))
                 + 1e3 * (states == 0))

    model = TracedModel(cv, time, states)

    for when, at in ((time, states), (time + 900.0, elsewhere)):
        assert _max_rel(model.rhs(when, at),
                        _numeric_rhs(cv, when, at)) < 1e-12, when


def test_iterated_tears_are_refused_under_a_trace(jax):
    """Tear iteration stops when the values say so, which a trace cannot."""
    from PharmaPy.ConcurrentVessel import ConcurrentVessel, StreamLink
    from PharmaPy.TracedBackends import TracedModel

    r01, r02 = _reactor(), _reactor(fed=False)
    cv = ConcurrentVessel({'R01': r01, 'R02': r02},
                          [StreamLink('R01', 'R02'),
                           StreamLink('R02', 'R01', fraction=0.5)])
    time, states = _start(cv)

    with pytest.raises(NotImplementedError, match="tear_mode='algebraic'"):
        TracedModel(cv, time, states)


def test_casadi_function_is_the_compiled_model(jax):
    ca = pytest.importorskip("casadi")
    from PharmaPy.TracedBackends import TracedModel

    unit = _crystallizer("continuous")
    time, states = _start(unit)
    model = TracedModel(unit, time, states)

    function = model.casadi_function()
    t = ca.MX.sym("t")
    x = ca.MX.sym("x", states.size)
    jacobian = ca.Function("J", [t, x], [ca.jacobian(function(t, x), x)])

    at = states + 1e-3 * np.max(states)

    assert np.array_equal(np.ravel(function(time, at).full()),
                          model.rhs(time, at))
    assert np.array_equal(np.asarray(jacobian(time, at).full()),
                          model.jac(time, at))
