"""Declared parameters (PharmaPy.Parameters) and the compiled model.

What is pinned here:

- Parameters.kinetics finds a unit's kinetics objects, a ConcurrentVessel's
  per member, and assign/values round-trip through set_params and
  concat_params (also for size-dependent growth, whose split used to be
  hard-coded).
- Declared parameters are inputs of the compiled model: after assigning new
  values the compiled right-hand side equals the unit's own, for each unit
  form, without a second trace.
- A solve after assigning new parameters gives what a fresh numeric solve
  gives, with one trace and no warning.
- The frozen-number bug: a number the trace froze in that changes
  undeclared (set_params, a controller setting) used to be silently ignored
  by the traced backends. Now the compiled model is checked before reuse,
  traced again with a warning, and the answer is right.

Everything that needs JAX skips without it.
"""

import warnings

import numpy as np
import pytest

from PharmaPy.IntegratorBackends import AssimuloBackend
from PharmaPy.Kinetics import CrystKinetics, RxnKinetics
from PharmaPy.Parameters import Parameters, find_kinetics
from PharmaPy.Phases_Refactored import LiquidPhase
from PharmaPy.ProcessControl_Refactored import ContinuousVesselController
from PharmaPy.Reactors_Refactored import ContinuousReactor
from PharmaPy.Streams_Refactored import LiquidStream

from test_traced_backends import (CONC_RXN, DATA_PATH, DIAMETER, DISSOL,
                                  EA_VALS, FEED_RXN, FEED_VOLFLOW, GROWTH,
                                  H_CONV, K_VALS, PRIM, RXNS, SEC, SOLUB,
                                  TEMP_RXN, VOLUME, _crystallizer, _max_rel,
                                  _numeric_rhs, _pair, _reactor, _start)

pytestmark = pytest.mark.unit

GRID = np.linspace(0.0, 3600.0, 13)  # [s]
FASTER = np.array([1.5, 1.5, 1.0, 1.0])  # k_1, k_2 up by half


def _reactor_with_orders(integrator=None):
    """The CSTR with its reaction orders as parameters too (params_f)."""
    unit = ContinuousReactor(
        integrator=integrator, h_conv=H_CONV, diam=DIAMETER,
        controller=ContinuousVesselController(temp_func=lambda t: TEMP_RXN))
    unit.Phases = LiquidPhase(DATA_PATH, temp=TEMP_RXN, mole_conc=CONC_RXN,
                              vol=VOLUME, name_solv='solvent')
    unit.RxnKinetics = RxnKinetics(path=DATA_PATH, rxn_list=RXNS,
                                   k_params=K_VALS, ea_params=EA_VALS,
                                   params_f=[[1.0, 1.0], [1.0, 1.0]])
    unit.Inlet = LiquidStream(DATA_PATH, temp=TEMP_RXN, mole_conc=FEED_RXN,
                              vol_flow=FEED_VOLFLOW, name_solv='solvent')
    return unit


def _final_conc(unit):
    return np.asarray(unit.result.mole_conc_liquid0)[-1]


# ---------------------------------------------------------------------
# No JAX needed
# ---------------------------------------------------------------------

def test_kinetics_are_found_and_round_trip():
    reactor = _reactor()
    parameters = Parameters.kinetics(reactor)

    assert parameters.size == 4
    assert parameters.names == ['k_1', 'k_2', 'E_{a, 1}', 'E_{a, 2}']
    assert parameters.free.all()

    values = parameters.values()
    parameters.assign(values * FASTER)
    assert np.allclose(find_kinetics(reactor)[0].concat_params(),
                       values * FASTER)

    parameters.free = [True, True, False, False]
    assert parameters.free_names == ['k_1', 'k_2']

    crystallizer = _crystallizer("batch")
    assert Parameters.kinetics(crystallizer).size == 13


def test_reaction_orders_are_parameters_when_given():
    parameters = Parameters.kinetics(_reactor_with_orders())

    assert parameters.size == 8
    values = parameters.values()

    parameters.assign(values + 0.1)
    assert np.allclose(parameters.values(), values + 0.1)


def test_size_dependent_growth_round_trips():
    """set_params split [3, 4, 3, 3] whatever the lengths, so with five
    growth parameters the last two landed in dissolution."""
    kinetics = CrystKinetics(SOLUB, nucl_prim=PRIM, nucl_sec=SEC,
                             growth=GROWTH + (0.1, 0.2), dissolution=DISSOL)
    values = np.asarray(kinetics.concat_params(), dtype=float)

    assert values.size == 15 == len(kinetics.name_params)

    kinetics.set_params(values * 2)
    assert np.allclose(kinetics.concat_params(), values * 2)
    assert np.allclose(kinetics.params['dissolution'], np.array(DISSOL) * 2)


def test_a_concurrent_vessel_declares_each_member():
    parameters = Parameters.kinetics(_pair(None))

    assert parameters.size == 8
    assert [block.name for block in parameters.blocks] == [
        'R01.RxnKinetics', 'R02.RxnKinetics']
    assert parameters.names[0] == 'R01.RxnKinetics:k_1'


def test_an_attribute_can_be_declared():
    controller = ContinuousVesselController(temp_func=lambda t: TEMP_RXN,
                                            target_volume=VOLUME)
    parameters = Parameters().add('target_volume', controller,
                                  'target_volume')

    assert parameters.names == ['target_volume']
    parameters.assign([0.05])
    assert controller.target_volume == 0.05
    assert type(controller.target_volume) is float


def test_an_owner_outside_the_unit_cannot_be_bound():
    parameters = Parameters().add('target_volume',
                                  ContinuousVesselController(
                                      temp_func=lambda t: TEMP_RXN,
                                      target_volume=VOLUME),
                                  'target_volume')

    with pytest.raises(ValueError, match="not part of the unit"):
        parameters.bind({})


def test_numeric_backends_read_parameters_live():
    pytest.importorskip("assimulo")

    unit = _reactor(integrator=AssimuloBackend(options={}))
    parameters = Parameters.kinetics(unit)
    values = parameters.values()

    unit.solve_unit(time_grid=GRID, verbose=False, parameters=parameters)
    parameters.assign(values * FASTER)
    unit.reset()
    unit.solve_unit(time_grid=GRID, verbose=False, parameters=parameters)

    fresh = _reactor(integrator=AssimuloBackend(options={}))
    Parameters.kinetics(fresh).assign(values * FASTER)
    fresh.solve_unit(time_grid=GRID, verbose=False)

    assert _max_rel(_final_conc(unit), _final_conc(fresh)) < 1e-10


# ---------------------------------------------------------------------
# JAX
# ---------------------------------------------------------------------

@pytest.fixture(scope="module")
def jax():
    return pytest.importorskip("jax")


UNITS = {
    "reactor": _reactor,
    "reactor_orders": _reactor_with_orders,
    "batch_fvm": lambda: _crystallizer("batch"),
    "continuous_moments": lambda: _crystallizer("continuous", "moments"),
    "pair": lambda: _pair(None),
}


@pytest.mark.parametrize("name", sorted(UNITS))
def test_declared_parameters_are_inputs_of_the_compiled_model(jax, name):
    from PharmaPy.TracedBackends import TracedModel

    unit = UNITS[name]()
    time, states = _start(unit)
    parameters = Parameters.kinetics(unit)
    values = parameters.values()

    model = TracedModel(unit, time, states, parameters)

    rng = np.random.default_rng(1)
    elsewhere = (states * (1 + 1e-2 * rng.standard_normal(states.size))
                 + 1e-6 * (states == 0))
    moved = values * (1 + 0.05 * np.sign(values))

    for new in (values, moved):
        parameters.assign(new)
        model.refresh_parameters()

        for when, at in ((time, states), (time + 600.0, elsewhere)):
            numeric = _numeric_rhs(unit, when, at)
            assert _max_rel(model.rhs(when, at), numeric) < 1e-12, when

    assert model.traces == 1


def test_exact_parameter_jacobian_matches_central_differences(jax):
    from PharmaPy.TracedBackends import TracedModel

    unit = _reactor()
    time, states = _start(unit)
    states = states + 0.05 * np.max(states)
    parameters = Parameters.kinetics(unit)
    values = parameters.values()

    model = TracedModel(unit, time, states, parameters)
    exact = model.jac_params(time, states)

    differences = np.empty_like(exact)

    for column in range(values.size):
        step = 1e-6 * abs(values[column])
        up, down = values.copy(), values.copy()
        up[column] += step
        down[column] -= step
        differences[:, column] = (model.rhs(time, states, up)
                                  - model.rhs(time, states, down)) / (2 * step)

    assert _max_rel(exact, differences) < 1e-7


def _traced_backend(name):
    from PharmaPy.TracedBackends import (CasadiBackend, JaxAssimuloBackend,
                                         JaxBackend)

    if name == "casadi":
        pytest.importorskip("casadi")
        return CasadiBackend(options={'maxh': 60})

    if name == "jax_cvode":
        return JaxAssimuloBackend(options={'maxh': 60})

    return JaxBackend(options={'maxh': 60})


def _truth(scale=FASTER, **controller):
    """AssimuloBackend's answer with the kinetic constants scaled."""
    pytest.importorskip("assimulo")

    unit = _reactor(integrator=AssimuloBackend(options={'maxh': 60}))
    parameters = Parameters.kinetics(unit)
    parameters.assign(parameters.values() * scale)

    for attribute, value in controller.items():
        setattr(unit.controller, attribute, value)

    unit.solve_unit(time_grid=GRID, verbose=False)

    return _final_conc(unit)


def _runtime_warnings(caught):
    return [w for w in caught if issubclass(w.category, RuntimeWarning)]


@pytest.mark.parametrize("backend", ["jax", "jax_cvode", "casadi"])
def test_new_parameter_values_need_no_new_trace(jax, backend):
    unit = _reactor(integrator=_traced_backend(backend))
    parameters = Parameters.kinetics(unit)

    unit.solve_unit(time_grid=GRID, verbose=False, parameters=parameters)
    model = unit.integrator.model

    parameters.assign(parameters.values() * FASTER)
    unit.reset()

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        unit.solve_unit(time_grid=GRID, verbose=False, parameters=parameters)

    assert not _runtime_warnings(caught)
    assert unit.integrator.model is model and model.traces == 1
    assert _max_rel(_final_conc(unit), _truth()) < 1e-4


@pytest.mark.parametrize("backend", ["jax", "jax_cvode", "casadi"])
def test_an_undeclared_change_is_traced_again_with_a_warning(jax, backend):
    """The frozen-number bug: set_params after compiling used to be
    ignored, C_C 0.11380 where the new constants give 0.11048."""
    unit = _reactor(integrator=_traced_backend(backend))
    unit.solve_unit(time_grid=GRID, verbose=False)
    first = unit.integrator.model

    kinetics = find_kinetics(unit)[0]
    kinetics.set_params(np.asarray(kinetics.concat_params()) * FASTER)
    unit.reset()

    with pytest.warns(RuntimeWarning, match="traced again"):
        unit.solve_unit(time_grid=GRID, verbose=False)

    assert unit.integrator.model is not first
    assert _max_rel(_final_conc(unit), _truth()) < 1e-4


def _set_point_run(integrator, declare):
    """A continuous reactor run for 1800 s, its volume set point lowered,
    and run on without a reset."""
    controller = ContinuousVesselController(temp_func=lambda t: TEMP_RXN,
                                            target_volume=VOLUME)
    unit = _reactor(integrator=integrator, controller=controller)

    parameters = (Parameters().add('target_volume', controller,
                                   'target_volume') if declare else None)

    unit.solve_unit(runtime=1800.0, verbose=False, parameters=parameters)
    controller.target_volume = 0.8 * VOLUME

    return unit, parameters


def _set_point_truth():
    pytest.importorskip("assimulo")

    truth, _ = _set_point_run(AssimuloBackend(options={'maxh': 60}), False)
    truth.solve_unit(runtime=1800.0, verbose=False)

    return np.asarray(truth.result.vessel_vol)[-1]


def test_a_controller_setting_changed_between_solves_is_seen(jax):
    """Not only kinetics: here the volume set point of a continuous
    reactor, changed part-way without a reset."""
    from PharmaPy.TracedBackends import JaxAssimuloBackend

    unit, _ = _set_point_run(JaxAssimuloBackend(options={'maxh': 60}), False)

    with pytest.warns(RuntimeWarning, match="traced again"):
        unit.solve_unit(runtime=1800.0, verbose=False)

    assert unit.result.time[0] == pytest.approx(1800.0)
    assert _max_rel(np.asarray(unit.result.vessel_vol)[-1],
                    _set_point_truth()) < 1e-4


def test_a_declared_controller_setting_needs_no_new_trace(jax):
    from PharmaPy.TracedBackends import JaxAssimuloBackend

    unit, parameters = _set_point_run(
        JaxAssimuloBackend(options={'maxh': 60}), True)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        unit.solve_unit(runtime=1800.0, verbose=False, parameters=parameters)

    assert not _runtime_warnings(caught)
    assert unit.integrator.model.traces == 1
    assert _max_rel(np.asarray(unit.result.vessel_vol)[-1],
                    _set_point_truth()) < 1e-4


def test_a_changed_declaration_compiles_again(jax):
    from PharmaPy.TracedBackends import JaxAssimuloBackend

    unit = _reactor(integrator=JaxAssimuloBackend(options={'maxh': 60}))
    parameters = Parameters.kinetics(unit)

    unit.solve_unit(time_grid=GRID, verbose=False, parameters=parameters)
    first = unit.integrator.model

    unit.solve_unit(time_grid=GRID, verbose=False)

    assert unit.integrator.model is not first
    assert unit.integrator.model.parameters is None
