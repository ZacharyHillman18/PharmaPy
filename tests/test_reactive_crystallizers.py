"""Reactive crystallizers (PharmaPy.Reactive_Crystallizers).

What is pinned here:

- Each form is a reactor and a crystallizer: both convenience initializers,
  the crystallizer's dilution and Krylov hint, the continuous one's default
  level controller.
- Its right-hand side is exactly that of the crystallizer with a
  ReactionMechanism added by hand -- the construction the class replaces --
  whichever kinetics is set first, with and without crystals present.
- The compiled right-hand side matches (JAX), and a continuous one runs.

Chemistry: A + B --> C in the solvent of compound_database.json, and C
crystallizes. Units: vol [m**3], vol_flow [m**3/s], mass_conc [kg/m**3],
temp [K], time [s].
"""

import numpy as np
import pytest

from PharmaPy.Crystallizers_Refactored import (BatchCrystallizer,
                                               ContinuousCrystallizer,
                                               SemiBatchCrystallizer)
from PharmaPy.DataClasses import IntraPhaseProcess, PhaseRef
from PharmaPy.IntegratorBackends import ScipyBackend
from PharmaPy.Kinetics import CrystKinetics, RxnKinetics
from PharmaPy.Mechanisms import OneDFVMMechanism, ReactionMechanism
from PharmaPy.Phases_Refactored import LiquidPhase, SolidPhase
from PharmaPy.ProcessControl_Refactored import (ContinuousVesselController,
                                                DefaultContinuousVesselVolume,
                                                SimpleTemperatureController)
from PharmaPy.Reactive_Crystallizers import (BatchReactiveCrystallizer,
                                             ContinuousReactiveCrystallizer,
                                             SemiBatchReactiveCrystallizer)
from PharmaPy.Reactors_Refactored import _BaseReactor
from PharmaPy.Streams_Refactored import LiquidStream

from test_traced_backends import (DATA_PATH, DISSOL, FEED_VOLFLOW, GROWTH,
                                  PRIM, SEC, SOLUB, TEMP_START, VOLUME, X_GRID,
                                  _max_rel, _numeric_rhs, _start, cooling)

pytestmark = pytest.mark.unit

RXN = ['A + B --> C']
K_RXN, EA_RXN = np.array([5e-2]), np.array([1e4])
CONC_AB = np.array([60.0, 60.0, 0.0, 0.0, 0.0])   # [kg/m3] reactants only
FEED_B = np.array([0.0, 80.0, 0.0, 0.0, 0.0])

FORMS = {
    "batch": (BatchReactiveCrystallizer, BatchCrystallizer),
    "semibatch": (SemiBatchReactiveCrystallizer, SemiBatchCrystallizer),
    "continuous": (ContinuousReactiveCrystallizer, ContinuousCrystallizer),
}


def _rxn():
    return RxnKinetics(path=DATA_PATH, rxn_list=RXN, k_params=K_RXN,
                       ea_params=EA_RXN)


def _cryst():
    return CrystKinetics(SOLUB, nucl_prim=PRIM, nucl_sec=SEC, growth=GROWTH,
                         dissolution=DISSOL)


def _controller(form):
    if form == "continuous":
        return ContinuousVesselController(temp_func=cooling)
    return SimpleTemperatureController(temp_func=cooling)


def _phases():
    solid = SolidPhase(DATA_PATH, mass=0.0, mass_frac=[0, 0, 1, 0, 0])
    solid.mechanisms = OneDFVMMechanism(
        solid, target_components='C', solvent_name='solvent', x_grid=X_GRID,
        distrib_init=np.zeros_like(X_GRID))
    liquid = LiquidPhase(DATA_PATH, temp=TEMP_START, mass_conc=CONC_AB,
                         vol=VOLUME, name_solv='solvent')
    return [liquid, solid]


def _feed(unit, form):
    if form != "batch":
        unit.Inlet = LiquidStream(DATA_PATH, temp=TEMP_START,
                                  mass_conc=FEED_B, vol_flow=FEED_VOLFLOW,
                                  name_solv='solvent')


def _reactive(form, reaction_first=True, integrator=None):
    cls = FORMS[form][0]
    unit = cls(integrator=integrator, controller=_controller(form))
    unit.Phases = _phases()
    if reaction_first:
        unit.RxnKinetics = _rxn()
        unit.CrystKinetics = _cryst()
    else:
        unit.CrystKinetics = _cryst()
        unit.RxnKinetics = _rxn()
    _feed(unit, form)
    return unit


def _by_hand(form):
    """The crystallizer with the reaction added by hand."""
    unit = FORMS[form][1](controller=_controller(form))
    unit.Phases = _phases()
    unit.CrystKinetics = _cryst()
    unit.intraphase_processes = [IntraPhaseProcess(
        phaseref=PhaseRef("liquid", 0),
        mechanism=ReactionMechanism(_rxn(), unit.Phases.Liquids[0]))]
    _feed(unit, form)
    return unit


def _with_crystals(unit, states):
    collection = unit.solver_state_collection
    distrib = collection.slices[next(key for key in collection.keys
                                     if key.name == "distrib")]
    cells = np.arange(distrib.stop - distrib.start)
    states = states.copy()
    states[distrib] = 1e6 * np.exp(-((cells - 6) / 3.0) ** 2)
    return states


@pytest.mark.parametrize("form", sorted(FORMS))
def test_each_form_is_a_reactor_and_a_crystallizer(form):
    unit = _reactive(form)

    assert isinstance(unit, _BaseReactor)
    assert isinstance(unit, FORMS[form][1])
    assert unit.oper_mode == FORMS[form][1].oper_mode
    assert unit.dilutes_intensive_states
    assert unit.linear_solver_hint == "krylov"
    assert len(unit.intraphase_processes) == 1
    assert len(unit.phase_connections) >= 1


def test_the_continuous_one_holds_its_level_by_default():
    unit = ContinuousReactiveCrystallizer()
    assert isinstance(unit.controller, DefaultContinuousVesselVolume)


@pytest.mark.parametrize("form", sorted(FORMS))
@pytest.mark.parametrize("reaction_first", [True, False])
def test_the_right_hand_side_is_the_hand_built_ones(form, reaction_first):
    unit, reference = _reactive(form, reaction_first), _by_hand(form)
    time, states = _start(unit)
    _start(reference)

    for at in (states, _with_crystals(unit, states)):
        expected = _numeric_rhs(reference, time, at)
        assert np.any(expected != 0)
        assert _max_rel(_numeric_rhs(unit, time, at), expected) < 1e-14


def test_the_compiled_right_hand_side_matches():
    pytest.importorskip("jax")
    from PharmaPy.TracedBackends import TracedModel

    unit = _reactive("continuous")
    time, states = _start(unit)
    crystals = _with_crystals(unit, states)
    model = TracedModel(unit, time, crystals)

    assert _max_rel(model.rhs(time, crystals),
                    _numeric_rhs(unit, time, crystals)) < 1e-12


def test_a_continuous_reactive_crystallizer_runs():
    """The feed's B reacts with the charge's A into C, which crystallizes."""
    unit = _reactive("continuous", integrator=ScipyBackend(
        options={'maxh': 60}))
    unit.solve_unit(runtime=1800.0, verbose=False)

    mass = np.asarray(unit.result.mass_j_liquid0)
    assert mass[-1, 0] < mass[0, 0]                      # A consumed
    assert np.max(np.asarray(unit.result.distrib_solid0)[-1]) > 0  # crystals
