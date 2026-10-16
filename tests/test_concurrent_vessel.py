"""Control across units in a ConcurrentVessel.

Two reactors in series, R01 -> R02, integrated as one system. The tests
check that a controller on the UPSTREAM unit can act on a measurement taken
in the DOWNSTREAM unit, directly or as the outer loop of a cascade, and that
the answer is a function of the state alone: evaluating the members in the
other order (which turns R01 -> R02 into a tear) must not change it, and the
results replay must report what the controller actually did.

Units: vol [m**3], vol_flow [m**3/s], mole_conc [kmol/m**3], temp [K],
time [s].
"""

import os

import numpy as np
import pytest

from PharmaPy.ConcurrentVessel import ConcurrentVessel, StreamLink
from PharmaPy.DataClasses import OperatingKey, PhaseRef
from PharmaPy.IntegratorBackends import ScipyBackend
from PharmaPy.Kinetics import RxnKinetics
from PharmaPy.Phases_Refactored import LiquidPhase
from PharmaPy.ProcessControl_Refactored import ContinuousVesselController
from PharmaPy.Reactors_Refactored import ContinuousReactor
from PharmaPy.Streams_Refactored import LiquidStream

pytestmark = pytest.mark.unit

DATA_PATH = os.path.join(
    os.path.dirname(__file__), "Flowsheet", "data", "compound_database.json"
)

# The reactors FlowsheetTester and ConcurrentTester use.
RXNS = ['A + B --> C', 'C + A --> D']
K_VALS = np.array([2.654e4, 5.3e2])
EA_VALS = np.array([4.0e4, 3.0e4])

TEMP = 313.15  # [K]
CONC_INIT = np.array([0.33, 0.33, 0, 0, 0])  # [kmol/m**3]
FEED_CONC = np.array([0.5, 0.5, 0, 0, 0])  # [kmol/m**3]
VOLUME = 0.06  # [m**3]
FEED_VOLFLOW = 1e-5  # [m**3/s]
H_CONV = 10000.0  # [W/(m**2 K)]
DIAMETER = 0.4  # [m]

C_INDEX = 2  # species C, the downstream measurement
RUN = 3 * 3600.0  # [s], about two residence times
GRID = np.linspace(0.0, RUN, 31)  # [s]

# R02's [C] after RUN with nothing controlled, recorded from this fixture.
C_OPEN_LOOP = 0.12136  # [kmol/m**3]

# Results reached by different evaluation orders agree to integration
# tolerance; anything order-dependent shows up orders of magnitude above it.
ORDER_RTOL = 1e-6  # [-]
REPLAY_RTOL = 1e-10  # [-]


def _reactor(controller, fed=True):
    """Return a continuous reactor, fed or not, with the given controller.

    Parameters
    ----------
    controller : Controller
        The reactor's single controller.
    fed : bool, optional
        Give it a fresh feed of FEED_CONC at FEED_VOLFLOW. A reactor fed
        through a link has none of its own.

    Returns
    -------
    ContinuousReactor
    """
    unit = ContinuousReactor(h_conv=H_CONV, diam=DIAMETER,
                             controller=controller)
    unit.Phases = LiquidPhase(DATA_PATH, temp=TEMP, mole_conc=CONC_INIT.copy(),
                              vol=VOLUME, name_solv='solvent')
    unit.RxnKinetics = RxnKinetics(path=DATA_PATH, rxn_list=RXNS,
                                   k_params=K_VALS, ea_params=EA_VALS)

    if fed:
        unit.Inlet = LiquidStream(DATA_PATH, temp=TEMP,
                                  mole_conc=FEED_CONC.copy(),
                                  vol_flow=FEED_VOLFLOW, name_solv='solvent')

    return unit


def _level_and_temperature():
    """The controller a plain continuous reactor here runs."""
    return ContinuousVesselController(temp_func=lambda t: TEMP)


def _pair(r01, r02, order=None):
    """Solve R01 -> R02 as one system and return the ConcurrentVessel."""
    cv = ConcurrentVessel({'R01': r01, 'R02': r02},
                          [StreamLink('R01', 'R02')],
                          integrator=ScipyBackend(options={'maxh': 60}),
                          order=order)
    cv.solve_unit(time_grid=GRID, verbose=False)

    return cv


def _final_c(unit):
    return float(np.asarray(unit.result.mole_conc_liquid0)[-1, C_INDEX])


def _order_gap(cv, other):
    return max(
        float(np.max(np.abs(np.asarray(cv[name].result.mass_j_liquid0)[-1]
                            - np.asarray(other[name].result.mass_j_liquid0)[-1])
                     / np.max(np.abs(np.asarray(
                         other[name].result.mass_j_liquid0)[-1]))))
        for name in cv.members)


# ---------------------------------------------------------------------
# An upstream feed set from a downstream measurement
# ---------------------------------------------------------------------
class FeedFromDownstream(ContinuousVesselController):
    """R01's controller: R01's feed from R02's [C]; R01's level as usual.

    It reads R02 while R01 is being evaluated. That is valid because a
    ConcurrentVessel brings every member to the evaluated state before any
    balance runs.
    """

    def __init__(self, downstream, c_set, gain=8.0):
        super().__init__(temp_func=lambda t: TEMP)

        self.downstream = downstream
        self.c_set = c_set
        self.gain = gain

    def actuate(self, time, completed_state, unit, resolved_inlets=None,
                resolved_outlets=None):

        if resolved_inlets is None:
            measured = float(self.downstream.Phases.Liquids[0].mole_conc[C_INDEX])
            error = (self.c_set - measured) / self.c_set

            key = OperatingKey('vol_flow', connection=0,
                               phaseref=PhaseRef('liquid', 0), port='inlet')
            self.operating_conditions[key] = (
                FEED_VOLFLOW * max(1.0 + self.gain * error, 0.05))
            return

        super().actuate(time, completed_state, unit, resolved_inlets,
                        resolved_outlets)


def _feed_controlled(order=None):
    r02 = _reactor(_level_and_temperature(), fed=False)
    r01 = _reactor(FeedFromDownstream(r02, c_set=1.1 * C_OPEN_LOOP))

    return _pair(r01, r02, order=order)


@pytest.fixture(scope='module')
def feed_controlled():
    return _feed_controlled()


def test_open_loop_reference_still_holds():
    """C_OPEN_LOOP is what an uncontrolled pair reaches; the tests rely on it."""
    cv = _pair(_reactor(_level_and_temperature()),
               _reactor(_level_and_temperature(), fed=False))

    assert _final_c(cv.R02) == pytest.approx(C_OPEN_LOOP, rel=1e-4)


def test_upstream_feed_moves_downstream_concentration(feed_controlled):
    """R02's [C] ends closer to its set point than it does uncontrolled."""
    c_set = 1.1 * C_OPEN_LOOP
    c_closed = _final_c(feed_controlled.R02)

    assert abs(c_closed - c_set) < 0.5 * abs(C_OPEN_LOOP - c_set)

    # ...because R01's feed, the manipulated variable, actually moved.
    feed = np.ravel(feed_controlled.R01.result.outlet_vol_flow)
    assert np.max(np.abs(feed - FEED_VOLFLOW)) > 0.1 * FEED_VOLFLOW


def test_upstream_control_does_not_depend_on_evaluation_order(feed_controlled):
    """Evaluating R02 first tears R01 -> R02 and must not change the answer."""
    reordered = _feed_controlled(order=['R02', 'R01'])

    assert [link.label for link in reordered.tears] == ['R01->R02']
    assert _order_gap(feed_controlled, reordered) < ORDER_RTOL


def test_replay_reports_what_the_upstream_controller_did(feed_controlled):
    """R01's reported outlet is what R02 received during the run.

    Replaying each member on its own handed R01's controller a frozen copy
    of R02, and R01's reported outlet came out up to 85% wrong.
    """
    reported = np.ravel(feed_controlled.R01.result.outlet_vol_flow)
    received = feed_controlled.result.couplings['R01->R02']['vol_flow']

    np.testing.assert_allclose(reported, received, rtol=REPLAY_RTOL)


# ---------------------------------------------------------------------
# Cascade: R02's controller sets the level R01's controller holds
# ---------------------------------------------------------------------
class QualityMaster(ContinuousVesselController):
    """R02's controller, and the outer loop of a cascade.

    It runs R02's own level and temperature, and on demand turns R02's [C]
    into the volume R01 should hold. It never writes into R01's controller.
    """

    def __init__(self, c_set, span=0.3, gain=10.0):
        super().__init__(temp_func=lambda t: TEMP)

        self.c_set = c_set
        self.span = span
        self.gain = gain
        self.vessel = None

    def setpoint_for(self, measured):
        error = (self.c_set - np.asarray(measured, dtype=float)) / self.c_set

        # More residence time in R01 turns C on into D, so a shortfall of C
        # asks for less volume upstream.
        return VOLUME * (1.0 - self.span * np.tanh(self.gain * error))

    def level_setpoint(self):
        return self.setpoint_for(
            self.vessel.Phases.Liquids[0].mole_conc[C_INDEX])


class LevelFollower(ContinuousVesselController):
    """R01's controller: a level loop that asks the master for its set point."""

    def __init__(self, master):
        self.master = master
        super().__init__(temp_func=lambda t: TEMP)

    @property
    def target_volume(self):
        return self.master.level_setpoint()

    @target_volume.setter
    def target_volume(self, value):
        pass  # the master owns this set point


def _cascade(order=None):
    master = QualityMaster(c_set=1.02 * C_OPEN_LOOP)

    r02 = _reactor(master, fed=False)
    master.vessel = r02

    r01 = _reactor(LevelFollower(master))

    return _pair(r01, r02, order=order)


@pytest.fixture(scope='module')
def cascade():
    return _cascade()


def test_cascade_inner_loop_follows_the_outer_set_point(cascade):
    """After start-up R01's level stays close to what R02's controller asks."""
    master = cascade.R02.controller

    requested = master.setpoint_for(
        np.asarray(cascade.R02.result.mole_conc_liquid0)[:, C_INDEX])
    held = np.ravel(cascade.R01.result.vessel_vol)

    excursion = np.max(np.abs(requested - VOLUME))
    assert excursion > 0.01 * VOLUME  # the outer loop did move it

    # R01 can only fill as fast as it is fed, so it trails a rising set
    # point during start-up; after that it must keep up.
    settled = np.asarray(cascade.result.time) > 0.25 * RUN
    assert np.max(np.abs(held - requested)[settled]) < 0.05 * excursion


def test_cascade_moves_downstream_concentration(cascade):
    c_set = 1.02 * C_OPEN_LOOP

    assert abs(_final_c(cascade.R02) - c_set) < abs(C_OPEN_LOOP - c_set)


def test_cascade_does_not_depend_on_evaluation_order(cascade):
    reordered = _cascade(order=['R02', 'R01'])

    assert _order_gap(cascade, reordered) < ORDER_RTOL


def test_cascade_replay_reports_what_the_inner_loop_did(cascade):
    reported = np.ravel(cascade.R01.result.outlet_vol_flow)
    received = cascade.result.couplings['R01->R02']['vol_flow']

    np.testing.assert_allclose(reported, received, rtol=REPLAY_RTOL)
