"""Flux limiters of the FVM population balance, OneDFVMMechanism(limiter=...).

What is pinned here:

- van Leer stays the default, and an unknown limiter is refused.
- Both limiters are second order on a smooth profile (phi(1) = 1) and drop
  the reconstruction where a profile is flat on one side (phi(0) = 0).
- The reason van Albada is offered: its limited slope is smooth across a
  sign change of the neighbouring differences, where van Leer's has a kink.
- With van Albada the compiled right-hand side equals the unit's own, and a
  solve conserves mass and lands close to van Leer's (the same equation,
  discretised a little differently).

Units: vol [m**3], mass [kg], temp [K], time [s].
"""

import numpy as np
import pytest

from PharmaPy.Crystallizers_Refactored import BatchCrystallizer
from PharmaPy.IntegratorBackends import AssimuloBackend
from PharmaPy.Kinetics import CrystKinetics
from PharmaPy.Mechanisms import (FLUX_LIMITERS, OneDFVMMechanism,
                                 van_albada_limiter, van_leer_limiter)
from PharmaPy.Phases_Refactored import LiquidPhase, SolidPhase
from PharmaPy.ProcessControl_Refactored import SimpleTemperatureController
from PharmaPy.Utilities import CoolingWater

from test_traced_backends import (CONC_C, DATA_PATH, DIAMETER, DISSOL, GROWTH,
                                  H_CONV, PRIM, SEC, SOLUB, TEMP_END,
                                  TEMP_START, VOLUME, X_GRID, _max_rel,
                                  _numeric_rhs, _start, cooling)

pytestmark = pytest.mark.unit


UNIFORM_GRID = np.arange(1.0, 400.0, 2.0)  # [micron]


def _batch(limiter=None, integrator=None, x_grid=X_GRID):
    """The FVM batch cooling crystallizer of test_traced_backends, with the
    limiter chosen (None: the mechanism's default)."""
    solid = SolidPhase(DATA_PATH, mass=0.0, mass_frac=[0, 0, 1, 0, 0])
    options = {} if limiter is None else {"limiter": limiter}
    solid.mechanisms = OneDFVMMechanism(
        solid, target_components='C', solvent_name='solvent',
        x_grid=x_grid, distrib_init=np.zeros_like(x_grid), **options)

    unit = BatchCrystallizer(
        integrator=integrator, h_conv=H_CONV, diam=DIAMETER,
        controller=SimpleTemperatureController(temp_func=cooling))
    unit.Phases = [LiquidPhase(DATA_PATH, temp=TEMP_START, mass_conc=CONC_C,
                               vol=VOLUME, name_solv='solvent'), solid]
    unit.CrystKinetics = CrystKinetics(SOLUB, nucl_prim=PRIM, nucl_sec=SEC,
                                       growth=GROWTH, dissolution=DISSOL)
    unit.Utility = CoolingWater(mass_flow=1, temp_in=TEMP_END)
    return unit


def _mechanism(unit):
    return next(m for m in unit.iter_mechanisms()
                if isinstance(m, OneDFVMMechanism))


def _with_a_humped_population(unit, states):
    """A population with a peak, so neighbouring differences change sign and
    both branches of each limiter are used. Small enough that growth does not
    drain the solute within the vessel's positivity horizon."""
    collection = unit.solver_state_collection
    distrib = collection.slices[next(key for key in collection.keys
                                     if key.name == "distrib")]
    cells = np.arange(distrib.stop - distrib.start)
    states = states.copy()
    states[distrib] = 1e6 * np.exp(-((cells - 6) / 3.0) ** 2)  # [#/(um m3)]
    return states


# ---------------------------------------------------------------------
# The limiters
# ---------------------------------------------------------------------

def test_van_leer_is_the_default_and_unknown_limiters_are_refused():
    assert _mechanism(_batch()).limiter == "van_leer"
    assert _mechanism(_batch("van_albada")).limiter == "van_albada"

    with pytest.raises(ValueError, match="not one of"):
        _batch("superbee")


@pytest.mark.parametrize("name", sorted(FLUX_LIMITERS))
def test_limiters_are_second_order_and_switch_off_at_a_flat_side(name):
    phi = FLUX_LIMITERS[name]

    assert phi(np.array(1.0)) == pytest.approx(1.0)
    assert phi(np.array(0.0)) == 0.0


def test_van_albada_is_smooth_where_van_leer_kinks():
    """The limited slope s(a, b) = phi(a/b) * b, across a = 0 (the left
    difference changing sign): one-sided derivatives agree for van Albada
    (both 1) and jump for van Leer (0 on the left, 2 on the right)."""
    b, h = 1.0, 1e-7

    def one_sided(phi):
        def slope(a):
            return phi(np.array(a / b)) * b
        return ((slope(0.0) - slope(-h)) / h, (slope(h) - slope(0.0)) / h)

    left, right = one_sided(van_albada_limiter)
    assert left == pytest.approx(right, abs=1e-6)

    left, right = one_sided(van_leer_limiter)
    assert left == pytest.approx(0.0, abs=1e-6)
    assert right == pytest.approx(2.0, abs=1e-6)


def test_the_limiter_changes_the_right_hand_side_only_with_crystals():
    """At an empty start both reconstructions are zero; with a population
    they differ, and the default is van Leer's."""
    default, leer, albada = _batch(), _batch("van_leer"), _batch("van_albada")
    time, states = _start(default)
    for unit in (leer, albada):
        _start(unit)

    assert np.array_equal(_numeric_rhs(leer, time, states),
                          _numeric_rhs(albada, time, states))

    crystals = _with_a_humped_population(default, states)
    assert np.array_equal(_numeric_rhs(default, time, crystals),
                          _numeric_rhs(leer, time, crystals))
    assert _max_rel(_numeric_rhs(albada, time, crystals),
                    _numeric_rhs(leer, time, crystals)) > 1e-6


# ---------------------------------------------------------------------
# Solves
# ---------------------------------------------------------------------

def test_van_albada_compiles_to_the_same_right_hand_side():
    pytest.importorskip("jax")
    from PharmaPy.TracedBackends import TracedModel

    unit = _batch("van_albada")
    time, states = _start(unit)
    crystals = _with_a_humped_population(unit, states)

    model = TracedModel(unit, time, crystals)

    assert _max_rel(model.rhs(time, crystals),
                    _numeric_rhs(unit, time, crystals)) < 1e-12


def test_van_albada_conserves_mass_and_tracks_van_leer():
    """On a uniform grid (the FVM assumes one) both limiters hold the vessel
    mass to 1e-4 and agree to 0.1 % on the moments: measured 9e-5 and
    1.3e-4 drift, moments within 0.09 %, liquid within 1e-6. On the geometric
    grid of the other tests both drift by about 1 %, from the grid."""
    pytest.importorskip("assimulo")

    finals = {}
    for name in ("van_leer", "van_albada"):
        unit = _batch(name, AssimuloBackend(options={'maxh': 60}),
                      x_grid=UNIFORM_GRID)
        unit.solve_unit(runtime=1200.0, verbose=False)
        result = unit.result

        total = np.asarray(result.Total_m_in_vessel, dtype=float)
        assert _max_rel(total, np.full_like(total, total[0])) < 1e-3, name
        assert np.min(result.distrib_solid0) > -1e-12, name

        finals[name] = (np.asarray(result.mass_j_liquid0)[-1],
                        np.asarray(result.mu_n)[-1])

    (liquid_va, mu_va), (liquid_vl, mu_vl) = (finals["van_albada"],
                                              finals["van_leer"])
    assert _max_rel(liquid_va, liquid_vl) < 1e-5
    for k in range(4):
        assert mu_va[k] == pytest.approx(mu_vl[k], rel=1e-2), k
