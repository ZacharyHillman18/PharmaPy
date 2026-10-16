"""What a vessel reports discharging is what its right-hand side discharged.

During the nucleation start-up of a continuous crystallizer the vessel's
positivity limiter throttles the outlet below what the level controller asks
for. The results replay used to resolve the outlet without the limiter, so
``result.outlet_vol_flow`` -- and ``outputs['vol_flow']``, which a flowsheet
hands to the next unit as its feed -- reported the request instead: up to 21%
above what actually left, during the first seconds.

Units: vol_flow [m**3/s], time [s], mass_conc [kg/m**3], temp [K],
x_grid [um], vol [m**3].
"""

import os

import numpy as np
import pytest

from PharmaPy.Crystallizers_Refactored import ContinuousCrystallizer
from PharmaPy.Kinetics import CrystKinetics
from PharmaPy.Mechanisms import OneDFVMMechanism
from PharmaPy.Phases_Refactored import LiquidPhase, SolidPhase
from PharmaPy.ProcessControl_Refactored import ContinuousVesselController
from PharmaPy.SimExec import SimulationExec
from PharmaPy.Streams_Refactored import LiquidStream
from PharmaPy.Utilities import CoolingWater

pytestmark = pytest.mark.unit

DATA_PATH = os.path.join(
    os.path.dirname(__file__), "Flowsheet", "data", "compound_database.json"
)

# The crystallizer FlowsheetTester and ConcurrentTester use.
TEMP = 278.15  # [K], below the 4.94 kg/m**3 solubility of C
CONC = np.array([0.0, 0.0, 40.0, 0.0, 0.0])  # [kg/m**3], supersaturated in C
VOLUME = 0.06  # [m**3]
FEED_VOLFLOW = 1e-5  # [m**3/s]
X_GRID = np.geomspace(1, 1500, num=35)  # [um]

SOLUBILITY = np.array([2.269e2, -1.88e0, 3.89e-3])
# Nucleation per m**3 of slurry, as the refactored mechanisms take it.
NUCL_PRIM = (3e8 * VOLUME, 0, 3)
NUCL_SEC = (4.46e10 * VOLUME, 0, 2, 1e-5)
GROWTH = (5, 0, 1.32)
DISSOLUTION = (1, 0, 1)

H_CONV = 10000.0  # [W/(m**2 K)]
DIAMETER = 0.4  # [m]

# The limiter engages within the first seconds, so a minute covers it.
RUNTIME = 60.0  # [s]

# Below this the test would not be exercising the limiter at all.
MIN_THROTTLING = 0.01  # [-]
REPORT_RTOL = 1e-10  # [-]


def _continuous_crystallizer():
    """Return the supersaturated, crystal-free continuous crystallizer.

    Returns
    -------
    ContinuousCrystallizer
        Fed with its own liquid at FEED_VOLFLOW, level held by its default
        volume loop, temperature held at TEMP.
    """
    unit = ContinuousCrystallizer(
        h_conv=H_CONV, diam=DIAMETER,
        controller=ContinuousVesselController(temp_func=lambda t: TEMP))

    solid = SolidPhase(DATA_PATH, mass=0.0, mass_frac=[0, 0, 1, 0, 0])
    solid.mechanisms = OneDFVMMechanism(
        solid, target_components='C', solvent_name='solvent',
        x_grid=X_GRID, distrib_init=np.zeros_like(X_GRID))

    unit.Phases = [LiquidPhase(DATA_PATH, temp=TEMP, mass_conc=CONC,
                               vol=VOLUME, name_solv='solvent'),
                   solid]
    unit.CrystKinetics = CrystKinetics(
        SOLUBILITY, nucl_prim=NUCL_PRIM, nucl_sec=NUCL_SEC, growth=GROWTH,
        dissolution=DISSOLUTION)
    unit.Inlet = LiquidStream(DATA_PATH, temp=TEMP, mass_conc=CONC,
                              vol_flow=FEED_VOLFLOW, name_solv='solvent')
    unit.Utility = CoolingWater(mass_flow=1, temp_in=TEMP)

    return unit


def _solver_states(unit):
    """Return the solved trajectory as solver vectors, one row per time.

    Parameters
    ----------
    unit : MultiPhaseVessel
        A solved vessel.

    Returns
    -------
    numpy.ndarray
        (n_times, n_states) array.
    """
    collection = unit.solver_state_collection
    time = np.asarray(unit.result.time)

    histories = {
        key: np.asarray(getattr(unit.result, collection.format_key(key)))
        for key in collection.keys
    }

    return np.array([
        collection.pack({key: values[i] for key, values in histories.items()})
        for i in range(time.size)
    ])


def _discharged_and_requested(unit, time, states):
    """Evaluate the right-hand side at each point; return what it discharged
    and what the level controller asked for.

    Parameters
    ----------
    unit : MultiPhaseVessel
        A solved vessel.
    time : numpy.ndarray
        Times [s].
    states : numpy.ndarray
        Solver vectors at those times.

    Returns
    -------
    tuple of numpy.ndarray
        (discharged, requested) total outlet flows [m**3/s].
    """
    discharged = np.empty(len(time))
    requested = np.empty(len(time))

    for i, (t, y) in enumerate(zip(time, states)):
        unit.unit_model(t, np.asarray(y, dtype=float))

        discharged[i] = unit.outlet_values()['vol_flow']

        point = unit._last_evaluation
        unthrottled = unit._resolve_outlets(point['completed_state'],
                                            point['operating_conditions'])
        requested[i] = sum(transfer.vol_flow
                           for connection in unthrottled
                           for transfer in connection)

    return discharged, requested


def test_reported_outlet_flow_is_what_left():
    """result.outlet_vol_flow equals the throttled flow, point by point."""
    unit = _continuous_crystallizer()
    time, states = unit.solve_unit(runtime=RUNTIME, verbose=False)

    discharged, requested = _discharged_and_requested(unit, time, states)

    # The check means something only if the limiter actually throttled.
    throttling = np.max(1.0 - discharged / requested)
    assert throttling > MIN_THROTTLING

    np.testing.assert_allclose(np.ravel(unit.result.outlet_vol_flow),
                               discharged, rtol=REPORT_RTOL)


def test_final_outlet_is_what_left():
    """outlet_conditions, behind unit.Outlet, is the throttled final outlet.

    The run is ended where the limiter throttles hardest, so the final point
    tells a throttled outlet from an unthrottled one. The transfers are
    compared rather than the outlet stream's phase volumes: a solid stream
    phase takes its volume from its size distribution, which differs from the
    flow the vessel draws whether or not the limiter is engaged.
    """
    probe = _continuous_crystallizer()
    time, states = probe.solve_unit(runtime=RUNTIME, verbose=False)

    discharged, requested = _discharged_and_requested(probe, time, states)
    hardest = float(time[np.argmax(1.0 - discharged / requested)])

    unit = _continuous_crystallizer()
    time, states = unit.solve_unit(runtime=hardest, verbose=False)

    discharged, requested = _discharged_and_requested(
        unit, time[-1:], states[-1:])

    assert 1.0 - discharged[0] / requested[0] > MIN_THROTTLING

    final = sum(transfer.vol_flow
                for connection in unit.outlet_conditions
                for transfer in connection)

    np.testing.assert_allclose(final, discharged[0], rtol=REPORT_RTOL)


def test_flowsheet_feeds_the_next_unit_what_left():
    """A downstream unit is fed the upstream unit's throttled discharge."""
    flst = SimulationExec(DATA_PATH, flowsheet='CR01 --> CR02')
    flst.CR01 = _continuous_crystallizer()
    flst.CR02 = _continuous_crystallizer()
    flst.SolveFlowsheet(
        kwargs_run={'CR01': {'runtime': RUNTIME, 'verbose': False},
                    'CR02': {'runtime': RUNTIME, 'verbose': False}},
        verbose=False)

    upstream = flst.CR01
    time = np.asarray(upstream.result.time)
    discharged, _ = _discharged_and_requested(
        upstream, time, _solver_states(upstream))

    # Connections writes the trajectory onto the wrapper or the inner phase
    # depending on what it was handed; look in both.
    stream = flst.CR02.inlet_connections[0].stream
    arrived = getattr(stream, 'y_inlet', None) or next(
        phase.y_inlet for phase in stream if getattr(phase, 'y_inlet', None))

    np.testing.assert_allclose(np.ravel(arrived['vol_flow']), discharged,
                               rtol=REPORT_RTOL)
