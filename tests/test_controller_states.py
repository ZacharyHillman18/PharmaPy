"""Controllers with solver states: PI(D) loops, and implicit control laws.

A controller declares solver states (Controller.solver_states) and the vessel
integrates them with everything else, so a loop's memory -- an integral, a
filtered measurement -- is a state: integrated on the accepted trajectory,
replayed, carried through a ConcurrentVessel, and traced. A state declared
'alg' is defined by a residual instead, for a control law that is implicit;
the vessel then needs a DAE backend.

What is pinned here:

- the PI volume loop holds the volume exactly where the proportional loop
  leaves an offset (the test liquid densifies as it reacts, so the outflow
  must differ from the inflow);
- the jacket PI loop tracks a temperature set point against a reaction heat
  load, and back-calculation keeps a saturated loop from winding up;
- a PI loop on one member of a ConcurrentVessel, measuring another member,
  removes the offset the proportional version leaves, also when traced;
- an algebraic controller state (an outlet set so that mass in = mass out)
  gives what the same law written explicitly gives, alone and inside a
  ConcurrentVessel, on IDA and on CasADi's IDAS.

Units: vol [m**3], vol_flow [m**3/s], mole_conc [kmol/m**3], temp [K],
time [s].
"""

import os

import numpy as np
import pytest

from PharmaPy import xp
from PharmaPy.DataClasses import (OperatingKey, PhaseRef, StateKey,
                                  StateVariable)
from PharmaPy.IntegratorBackends import AssimuloBackend, ScipyBackend
from PharmaPy.Kinetics import RxnKinetics
from PharmaPy.Phases_Refactored import LiquidPhase
from PharmaPy.ProcessControl_Refactored import (
    ContinuousVesselController, Controller,
    JacketPIDContinuousVesselController, PIDContinuousVesselController,
    PIDLoop)
from PharmaPy.Reactors_Refactored import ContinuousReactor
from PharmaPy.Streams_Refactored import LiquidStream
from PharmaPy.Utilities import CoolingWater

pytestmark = pytest.mark.unit

DATA_PATH = os.path.join(
    os.path.dirname(__file__), "Flowsheet", "data", "compound_database.json"
)

# The continuous reactor of FlowsheetTester and test_concurrent_vessel.py.
# Its species densities differ (A 1230, B 865, C 1200, D 867 kg/m**3), so the
# liquid's density changes as it reacts.
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

RUN = 3 * 3600.0  # [s], about two residence times
GRID = np.linspace(0.0, RUN, 31)  # [s]


def _reactor(controller, integrator=None, fed=True, utility=False,
             delta_hrxn=None):
    unit = ContinuousReactor(
        integrator=integrator or AssimuloBackend(options={'maxh': 60}),
        controller=controller, h_conv=H_CONV, diam=DIAMETER)
    unit.Phases = LiquidPhase(DATA_PATH, temp=TEMP, mole_conc=CONC_INIT,
                              vol=VOLUME, name_solv='solvent')
    kwargs = {} if delta_hrxn is None else {"delta_hrxn": delta_hrxn}
    unit.RxnKinetics = RxnKinetics(path=DATA_PATH, rxn_list=RXNS,
                                   k_params=K_VALS, ea_params=EA_VALS,
                                   **kwargs)

    if fed:
        unit.Inlet = LiquidStream(DATA_PATH, temp=TEMP, mole_conc=FEED_CONC,
                                  vol_flow=FEED_VOLFLOW, name_solv='solvent')

    if utility:
        unit.Utility = CoolingWater(mass_flow=1.0, temp_in=TEMP)

    return unit


def _volume_offset(unit):
    volume = np.ravel(unit.result.vessel_vol)
    return volume[-1] / VOLUME - 1


# ---------------------------------------------------------------------
# PI volume loop
# ---------------------------------------------------------------------

def test_pi_volume_loop_removes_the_proportional_offset():
    proportional = _reactor(ContinuousVesselController(
        temp_func=lambda t: TEMP))
    proportional.solve_unit(time_grid=GRID, verbose=False)

    integral = _reactor(PIDContinuousVesselController(
        temp_func=lambda t: TEMP))
    integral.solve_unit(time_grid=GRID, verbose=False)

    # The proportional loop needs an error to hold an outflow different
    # from the inflow; the PI loop's integral carries that instead.
    assert abs(_volume_offset(proportional)) > 1e-5
    assert abs(_volume_offset(integral)) < 1e-3 * abs(
        _volume_offset(proportional))


def test_the_integral_is_a_solver_state_and_is_reported():
    unit = _reactor(PIDContinuousVesselController(temp_func=lambda t: TEMP))
    unit.solve_unit(time_grid=GRID, verbose=False)

    assert StateKey("level_integral") in unit.solver_state_collection.keys

    integral = np.ravel(unit.result.level_integral)

    assert integral.shape == GRID.shape
    assert integral[0] == 0.0
    # It settles where Kc * I / Ti carries the outflow's departure from the
    # inflow, so it ends away from zero and still.
    assert abs(integral[-1]) > 0
    assert abs(integral[-1] - integral[-2]) < 1e-3 * abs(integral[-1])


def test_derivative_action_adds_a_filtered_measurement():
    controller = PIDContinuousVesselController(
        temp_func=lambda t: TEMP, derivative_time=60.0)
    unit = _reactor(controller)
    unit.solve_unit(time_grid=GRID, verbose=False)

    keys = unit.solver_state_collection.keys

    assert StateKey("level_filtered") in keys
    # The filter starts at the measured volume, so there is no kick.
    assert np.ravel(unit.result.level_filtered)[0] == pytest.approx(VOLUME)
    assert abs(_volume_offset(unit)) < 1e-6


# ---------------------------------------------------------------------
# PI temperature loop through the jacket
# ---------------------------------------------------------------------

STEP_TIME = 3600.0  # [s]
STEP_TO = 303.15  # [K]
HEAT_OF_REACTION = [-8e4, -6e4]  # [J/mol], exothermic: a load to reject


def _step(time):
    return TEMP if time < STEP_TIME else STEP_TO


def _temperature_errors(integral_time=None):
    """Temperature error just before the set point step, and at the end."""
    unit = _reactor(
        JacketPIDContinuousVesselController(
            temp_func=_step,
            temperature={"jacket_limits": (273.15, 373.15),
                         "integral_time": integral_time}),
        utility=True, delta_hrxn=HEAT_OF_REACTION)
    unit.solve_unit(time_grid=GRID, verbose=False)

    time = unit.result.time
    temp = np.ravel(unit.result.global_temp)

    return (abs(temp[time <= STEP_TIME][-1] - TEMP), abs(temp[-1] - STEP_TO),
            unit)


def test_jacket_pi_loop_tracks_a_set_point_step_under_load():
    before, end, unit = _temperature_errors()

    # A proportional loop alone holds the reaction heat with a steady error
    # (integral_time far longer than the run switches the integral off).
    p_before, p_end, _ = _temperature_errors(integral_time=1e12)

    assert p_before > 0.05 and p_end > 0.05
    assert before < 1e-3 and end < 1e-4
    assert abs(_volume_offset(unit)) < 1e-6


def _overshoot_after_saturation(tracking_time):
    # A jacket that cannot reach the new set point at first saturates; with
    # a narrow range the loop sits at its limit for a while.
    unit = _reactor(
        JacketPIDContinuousVesselController(
            temp_func=_step,
            temperature={"jacket_limits": (300.15, 373.15),
                         "integral_time": 600.0,
                         "tracking_time": tracking_time}),
        utility=True, delta_hrxn=HEAT_OF_REACTION)
    unit.solve_unit(time_grid=np.linspace(0, RUN, 301), verbose=False)

    time = unit.result.time
    temp = np.ravel(unit.result.global_temp)

    return np.max(STEP_TO - temp[time > STEP_TIME])


def test_back_calculation_limits_windup_at_the_jacket_limit():
    with_tracking = _overshoot_after_saturation(tracking_time=None)
    without = _overshoot_after_saturation(tracking_time=1e12)

    assert with_tracking < without


# ---------------------------------------------------------------------
# A PI loop across two members of a ConcurrentVessel
# ---------------------------------------------------------------------

# The loop holds R02's reactant A, which R01's feed moves strongly. (R02's
# intermediate C, which test_concurrent_vessel.py controls, sits near its
# maximum over feed rate at these conditions, so a loop on it converges over
# a day rather than a few hours.)
A_INDEX = 0
A_SET = 0.0045  # [kmol/m**3], about 1.2 times R02's open-loop value


class PIFeedFromDownstream(ContinuousVesselController):
    """R01's feed flow from R02's [A], with integral action.

    Shaped like test_concurrent_vessel.FeedFromDownstream, the proportional
    version. The integral is a solver state of R01, the vessel that owns the
    controller, although the measurement is R02's. Written to the tracing
    rules (xp.maximum, no float()).
    """

    FEED_KEY = OperatingKey('vol_flow', connection=0,
                            phaseref=PhaseRef('liquid', 0), port='inlet')

    def __init__(self, downstream, gain=8.0, integral_time=3600.0,
                 integral=True):
        super().__init__(temp_func=lambda t: TEMP)

        self.downstream = downstream
        self.gain = gain
        self.integral_time = integral_time
        self.feed_loop = PIDLoop("feed", action="reverse", integral=integral)
        self.feed_integral = 0.0

    @property
    def solver_states(self):
        return tuple(super().solver_states) + self.feed_loop.solver_states

    def _law(self, completed_state):
        measured = self.downstream.Phases.Liquids[0].mole_conc[A_INDEX] / A_SET
        law = self.feed_loop.law(completed_state, 1.0, measured, self.gain,
                                 self.integral_time, bias=1.0)
        return measured, law, xp.maximum(law, 0.05)

    def actuate(self, time, completed_state, unit, resolved_inlets=None,
                resolved_outlets=None):

        if resolved_inlets is None:
            self.operating_conditions[self.FEED_KEY] = (
                FEED_VOLFLOW * self._law(completed_state)[2])
            return

        super().actuate(time, completed_state, unit, resolved_inlets,
                        resolved_outlets)

    def get_solver_state_rates(self, time, completed_state, unit,
                               resolved_inlets=None, resolved_outlets=None,
                               operating_conditions=None):

        rates = dict(super().get_solver_state_rates(
            time, completed_state, unit, resolved_inlets, resolved_outlets,
            operating_conditions))

        measured, law, applied = self._law(completed_state)
        rates.update(self.feed_loop.rates(
            completed_state, 1.0, measured, law, applied, self.gain,
            self.integral_time))

        return rates


def _cross_unit_pair(integrator, integral=True):
    from PharmaPy.ConcurrentVessel import ConcurrentVessel, StreamLink

    r02 = _reactor(ContinuousVesselController(temp_func=lambda t: TEMP),
                   fed=False)
    r01 = _reactor(PIFeedFromDownstream(r02, integral=integral))

    return ConcurrentVessel({'R01': r01, 'R02': r02},
                            [StreamLink('R01', 'R02')],
                            integrator=integrator)


def _final_a(cv):
    return float(np.asarray(cv['R02'].result.mole_conc_liquid0)[-1, A_INDEX])


LONG_GRID = np.linspace(0.0, 12 * 3600.0, 49)  # [s]


def test_pi_loop_across_members_removes_the_offset():
    proportional = _cross_unit_pair(ScipyBackend(options={'maxh': 60}),
                                    integral=False)
    proportional.solve_unit(time_grid=LONG_GRID, verbose=False)

    integral = _cross_unit_pair(ScipyBackend(options={'maxh': 60}))
    integral.solve_unit(time_grid=LONG_GRID, verbose=False)

    p_error = abs(_final_a(proportional) / A_SET - 1)
    pi_error = abs(_final_a(integral) / A_SET - 1)

    assert p_error > 5e-3
    assert pi_error < 0.05 * p_error
    assert StateKey("feed_integral") in integral['R01'].solver_state_collection.keys


@pytest.mark.parametrize("backend", ["jax", "casadi"])
def test_pi_loop_across_members_traces(backend):
    pytest.importorskip("jax")
    from PharmaPy.TracedBackends import CasadiBackend, JaxBackend

    if backend == "casadi":
        pytest.importorskip("casadi")
        integrator = CasadiBackend(options={'maxh': 60})
    else:
        integrator = JaxBackend(options={'maxh': 60})

    expected = _cross_unit_pair(ScipyBackend(options={'maxh': 60}))
    expected.solve_unit(time_grid=LONG_GRID, verbose=False)

    traced = _cross_unit_pair(integrator)
    traced.solve_unit(time_grid=LONG_GRID, verbose=False)

    assert _final_a(traced) == pytest.approx(_final_a(expected), rel=1e-5)


def test_traced_rhs_with_controller_states_matches_the_unit():
    pytest.importorskip("jax")
    from PharmaPy.TracedBackends import TracedModel

    unit = _reactor(
        JacketPIDContinuousVesselController(
            temp_func=_step, derivative_time=60.0,
            temperature={"jacket_limits": (273.15, 373.15),
                         "derivative_time": 30.0}),
        utility=True, delta_hrxn=HEAT_OF_REACTION)
    unit.compile_structure()
    unit.reset()
    states = np.asarray(unit.create_solver_init_states(), dtype=float)
    model = TracedModel(unit, 0.0, states)

    # Either side of the set point step, away from the start state.
    for time in (0.0, 1.5 * STEP_TIME):
        at = states * 1.01
        numeric = np.array(unit.unit_model(time, at), dtype=float)
        compiled = model.rhs(time, at)

        assert np.max(np.abs(compiled - numeric)) < 1e-12 * np.max(
            np.abs(numeric))


# ---------------------------------------------------------------------
# An algebraic controller state: an implicit control law
# ---------------------------------------------------------------------

def _mass_in(unit, resolved_inlets):
    return sum(transfer.stream_phase.mass_flow
               for connection in resolved_inlets
               for transfer in connection)


class MassBalancedOutlet(Controller):
    """Hold the vessel's mass: what leaves equals what arrives, by mass.

    The outlet's volumetric flow is an algebraic solver state, defined by

        (rho * Q_out - m_in) / m_ref = 0

    rather than computed as m_in / rho. It is stored scaled by the feed, so
    the solver's tolerances mean the same for it as for everything else.
    """

    solver_states = (
        StateVariable(name="outlet_flow_scaled", dim=1, units="-",
                      state_type="alg", limit_negative_inventory=False),
    )

    KEY = StateKey("outlet_flow_scaled")
    OUTLET = OperatingKey("vol_flow", connection=0, port="outlet")

    def initial_solver_state(self, state_key, unit=None):
        # Consistent at the start: the feed's mass over the liquid density.
        mass_in = sum(phase.mass_flow
                      for connection in unit.inlet_connections
                      for phase in connection.stream)
        return mass_in / unit.Phases.Liquids[0].density / FEED_VOLFLOW

    def actuate(self, time, completed_state, unit, resolved_inlets=None,
                resolved_outlets=None):

        if resolved_inlets is None:
            return

        self.operating_conditions[self.OUTLET] = (
            completed_state[self.KEY] * FEED_VOLFLOW)

    def get_solver_state_residuals(self, time, completed_state, unit,
                                   resolved_inlets=None,
                                   resolved_outlets=None,
                                   operating_conditions=None):

        mass_in = _mass_in(unit, resolved_inlets)
        mass_out = (unit.Phases.Liquids[0].density
                    * completed_state[self.KEY] * FEED_VOLFLOW)

        return {self.KEY: (mass_out - mass_in) / (FEED_VOLFLOW * 1000.0)}


class ExplicitMassBalancedOutlet(Controller):
    """The same law, solved for the outlet by hand: Q_out = m_in / rho."""

    def actuate(self, time, completed_state, unit, resolved_inlets=None,
                resolved_outlets=None):

        if resolved_inlets is None:
            return

        self.operating_conditions[MassBalancedOutlet.OUTLET] = (
            _mass_in(unit, resolved_inlets)
            / unit.Phases.Liquids[0].density)


def _final_mass_j(unit):
    return np.asarray(unit.result.mass_j_liquid0)[-1]


@pytest.fixture(scope="module")
def explicit_mass_balanced():
    unit = _reactor(ExplicitMassBalancedOutlet())
    unit.solve_unit(time_grid=GRID, verbose=False)
    return unit


def test_algebraic_controller_state_matches_its_explicit_form(
        explicit_mass_balanced):
    pytest.importorskip("assimulo")
    from PharmaPy.IntegratorBackends import AssimuloDAEBackend

    unit = _reactor(MassBalancedOutlet(),
                    integrator=AssimuloDAEBackend(options={'maxh': 60}))
    unit.solve_unit(time_grid=GRID, verbose=False)

    assert unit.has_algebraic_balance
    expected = _final_mass_j(explicit_mass_balanced)
    assert np.max(np.abs(_final_mass_j(unit) - expected)) < 1e-5 * np.max(
        np.abs(expected))

    # The mass held is the charge's: in = out at every point.
    total = np.ravel(unit.result.Total_m_in_vessel)
    assert np.max(np.abs(total / total[0] - 1)) < 1e-6


def test_algebraic_controller_state_on_casadi_idas(explicit_mass_balanced):
    pytest.importorskip("jax")
    pytest.importorskip("casadi")
    from PharmaPy.TracedBackends import CasadiBackend

    unit = _reactor(MassBalancedOutlet(),
                    integrator=CasadiBackend(options={'maxh': 60}))
    unit.solve_unit(time_grid=GRID, verbose=False)

    expected = _final_mass_j(explicit_mass_balanced)
    assert np.max(np.abs(_final_mass_j(unit) - expected)) < 1e-5 * np.max(
        np.abs(expected))


def test_an_ode_backend_refuses_an_algebraic_controller_state():
    unit = _reactor(MassBalancedOutlet(),
                    integrator=ScipyBackend(options={'maxh': 60}))

    with pytest.raises(NotImplementedError, match="algebraic"):
        unit.solve_unit(time_grid=GRID, verbose=False)


def test_algebraic_controller_state_inside_a_concurrent_vessel():
    pytest.importorskip("assimulo")
    from PharmaPy.ConcurrentVessel import ConcurrentVessel, StreamLink
    from PharmaPy.IntegratorBackends import AssimuloDAEBackend

    def pair(controller, integrator):
        r01 = _reactor(controller)
        r02 = _reactor(ContinuousVesselController(temp_func=lambda t: TEMP),
                       fed=False)
        cv = ConcurrentVessel({'R01': r01, 'R02': r02},
                              [StreamLink('R01', 'R02')],
                              integrator=integrator)
        cv.solve_unit(time_grid=GRID, verbose=False)
        return cv

    implicit = pair(MassBalancedOutlet(),
                    AssimuloDAEBackend(options={'maxh': 60}))
    explicit = pair(ExplicitMassBalancedOutlet(),
                    AssimuloBackend(options={'maxh': 60}))

    assert implicit.has_algebraic_balance

    for name in ('R01', 'R02'):
        got = _final_mass_j(implicit[name])
        expected = _final_mass_j(explicit[name])
        assert np.max(np.abs(got - expected)) < 1e-5 * np.max(
            np.abs(expected)), name
