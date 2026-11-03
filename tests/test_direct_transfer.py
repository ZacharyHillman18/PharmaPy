"""DirectTransfer, and phase connections without kinetics.

A PhaseConnection names two phases; its mechanism says how material moves
between them and owns whatever rate law that takes. DirectTransfer is the
simplest such mechanism: a rate, given as a callable or a constant.

The fixture is a batch vessel holding two liquid phases of the same
species, so every number can be checked by hand. Units: mass [kg], rate
[kg/s], time [s], temp [K].
"""

import os

import numpy as np
import pytest

from PharmaPy.DataClasses import PhaseConnection, PhaseRef, StateKey
from PharmaPy.IntegratorBackends import ScipyBackend
from PharmaPy.Mechanisms import DirectTransfer
from PharmaPy.Phases_Refactored import LiquidPhase
from PharmaPy.Reactors_Refactored import BatchReactor
from PharmaPy.Utilities import CoolingWater

pytestmark = pytest.mark.unit

DATA_PATH = os.path.join(
    os.path.dirname(__file__), "Flowsheet", "data", "compound_database.json"
)

C_INDEX, SOLVENT_INDEX = 2, 4
MASS_0, MASS_1 = 1.0, 0.5  # [kg], the two liquids
FRAC_0 = [0, 0, 0.3, 0, 0.7]  # C in solvent
FRAC_1 = [0, 0, 0, 0, 1.0]  # solvent only
TEMP = 298.15  # [K]

TIGHT = {'atol': 1e-12, 'rtol': 1e-10}
GRID = np.linspace(0.0, 3600.0, 61)  # [s]

SOURCE, SINK = PhaseRef('liquid', 0), PhaseRef('liquid', 1)


def _vessel(mechanism, integrator=None, isothermal=True, weights=None):
    unit = BatchReactor(
        integrator=integrator or ScipyBackend(options=dict(TIGHT)),
        isothermal=isothermal, h_conv=1000.0, diam=0.1)
    unit.Phases = [
        LiquidPhase(DATA_PATH, temp=TEMP, mass_frac=FRAC_0, mass=MASS_0,
                    name_solv='solvent'),
        LiquidPhase(DATA_PATH, temp=TEMP, mass_frac=FRAC_1, mass=MASS_1,
                    name_solv='solvent'),
    ]
    unit.phase_connections = [
        PhaseConnection(SOURCE, SINK, species_weights=weights,
                        mechanism=mechanism)]
    return unit


def _masses(unit):
    return (np.asarray(unit.result.mass_j_liquid0),
            np.asarray(unit.result.mass_j_liquid1))


def test_a_phase_connection_needs_no_kinetics():
    connection = PhaseConnection(SOURCE, SINK,
                                 mechanism=DirectTransfer(0.0))

    assert not hasattr(connection, "kinetics")

    with pytest.raises(TypeError):
        PhaseConnection(SOURCE, SINK, kinetics=object())


def test_first_order_transfer_matches_the_exact_solution():
    rate_constant = 1e-3  # [1/s]
    unit = _vessel(DirectTransfer.first_order(rate_constant, species=['C']))
    unit.solve_unit(time_grid=GRID, verbose=False)

    source, sink = _masses(unit)
    c_initial = FRAC_0[C_INDEX] * MASS_0
    exact = c_initial * np.exp(-rate_constant * GRID)

    assert np.max(np.abs(source[:, C_INDEX] - exact)) < 1e-7 * c_initial
    assert np.max(np.abs(sink[:, C_INDEX] - (c_initial - exact))) < (
        1e-7 * c_initial)

    # Only C moves; the solvent stays where it was, and nothing is lost.
    assert np.allclose(source[:, SOLVENT_INDEX], FRAC_0[SOLVENT_INDEX] * MASS_0)
    assert np.allclose(sink[:, SOLVENT_INDEX], MASS_1)
    total = source.sum(axis=1) + sink.sum(axis=1)
    assert np.max(np.abs(total - (MASS_0 + MASS_1))) < 1e-12


def test_a_scalar_rate_moves_the_source_as_it_is():
    rate = 1e-5  # [kg/s], total
    unit = _vessel(DirectTransfer(rate))
    unit.solve_unit(time_grid=GRID, verbose=False)

    source, sink = _masses(unit)

    # The source keeps its composition, so each species leaves in its
    # proportion: here C 30 %, solvent 70 %.
    moved = sink[-1] - sink[0]
    assert moved.sum() == pytest.approx(rate * GRID[-1], rel=1e-8)
    assert moved[C_INDEX] / moved.sum() == pytest.approx(0.3, rel=1e-8)


def test_species_weights_split_a_scalar_rate():
    rate = 1e-5  # [kg/s]
    weights = np.array([0, 0, 1.0, 0, 0])
    unit = _vessel(DirectTransfer(rate), weights=weights)
    unit.solve_unit(time_grid=GRID, verbose=False)

    source, sink = _masses(unit)
    moved = sink[-1] - sink[0]

    assert moved[C_INDEX] == pytest.approx(rate * GRID[-1], rel=1e-8)
    assert moved[SOLVENT_INDEX] == pytest.approx(0.0, abs=1e-12)


def test_a_negative_rate_flows_back():
    flows = np.zeros(5)
    flows[SOLVENT_INDEX] = -2e-5  # [kg/s], solvent from sink to source
    unit = _vessel(DirectTransfer(flows))
    unit.solve_unit(time_grid=GRID, verbose=False)

    source, sink = _masses(unit)

    assert source[-1, SOLVENT_INDEX] - source[0, SOLVENT_INDEX] == (
        pytest.approx(2e-5 * GRID[-1], rel=1e-8))


def test_a_transfer_cannot_drain_a_phase_below_zero():
    # Enough to remove all of C in a fifth of the run.
    flows = np.zeros(5)
    flows[C_INDEX] = FRAC_0[C_INDEX] * MASS_0 / (0.2 * GRID[-1])
    unit = _vessel(DirectTransfer(flows))
    unit.solve_unit(time_grid=GRID, verbose=False)

    source, sink = _masses(unit)

    # The positivity limiter throttles it as C runs out.
    assert source[:, C_INDEX].min() > -1e-9
    assert sink[-1, C_INDEX] == pytest.approx(FRAC_0[C_INDEX] * MASS_0,
                                              rel=1e-4)


def test_the_heat_of_transfer_enters_the_energy_balance():
    rate = 1e-4  # [kg/s]
    heat = -2e5  # [J/kg], exothermic

    def build(heat_of_transfer):
        unit = _vessel(DirectTransfer(rate, heat_of_transfer=heat_of_transfer),
                       isothermal=False)
        unit.Utility = CoolingWater(mass_flow=1.0, temp_in=TEMP)
        unit.compile_structure()
        unit.reset()
        states = np.asarray(unit.create_solver_init_states(), dtype=float)
        rates = np.asarray(unit.unit_model(0.0, states), dtype=float)
        temp_slot = unit.solver_state_collection.slices[StateKey("global_temp")]
        return unit, rates[temp_slot][0]

    unit, with_heat = build(heat)
    _, without = build(0.0)

    heat_capacity = unit.Phases.getCp(basis='mass')
    assert with_heat - without == pytest.approx(-heat * rate / heat_capacity,
                                                rel=1e-10)


def test_a_connection_without_a_mechanism_is_refused():
    unit = _vessel(mechanism=None)

    with pytest.raises(ValueError, match="has no mechanism"):
        unit.solve_unit(time_grid=GRID, verbose=False)


def test_direct_transfer_traces():
    pytest.importorskip("jax")
    from PharmaPy.TracedBackends import JaxBackend

    expected = _vessel(DirectTransfer.first_order(1e-3, species=['C']))
    expected.solve_unit(time_grid=GRID, verbose=False)

    traced = _vessel(DirectTransfer.first_order(1e-3, species=['C']),
                     integrator=JaxBackend(options=dict(TIGHT)))
    traced.solve_unit(time_grid=GRID, verbose=False)

    assert np.max(np.abs(_masses(traced)[0] - _masses(expected)[0])) < 1e-7
