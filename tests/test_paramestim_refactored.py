"""Parameter estimation on the refactored units (ParamEstim_Refactored).

What is pinned here:

- Experiment hands legacy ParameterEstimation the measured columns and their
  sensitivities in the layout legacy CVODES runs did (stacked column by
  column).
- Parameters used to generate synthetic data are recovered from a perturbed
  start: the CSTR (k_1, k_2) and the FVM batch crystallizer (growth k_g, g).
- The Ziegler case of doc/online_docs/examples/param_estimation_solved.ipynb
  (A -> B, A -> C, B -> C, B -> D at five temperatures, B and C measured,
  second stage) gives legacy PharmaPy's estimates on the refactored
  BatchReactor, and legacy StatsModule runs on the result.

Units: time [s], mole_conc [kmol/m**3 = mol/L], mass_conc [kg/m**3].
"""

import json
import os

import numpy as np
import pytest

from PharmaPy.IntegratorBackends import AssimuloBackend
from PharmaPy.Kinetics import RxnKinetics
from PharmaPy.ParamEstim_Refactored import Experiment, ParameterEstimation
from PharmaPy.Parameters import Parameters
from PharmaPy.Phases_Refactored import LiquidPhase
from PharmaPy.Reactors_Refactored import BatchReactor

from test_traced_backends import _crystallizer, _reactor

pytestmark = pytest.mark.unit

ZIEGLER = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                       "workshop_data", "Case_study_1")
ZIEGLER_PATH = os.path.join(ZIEGLER, "pure_components_ziegler.json")
ZIEGLER_RXNS = ['A --> B', 'A --> C', 'B --> C', 'B --> D']
ZIEGLER_TEMPS = [673, 698, 723, 748, 773]  # [K]

# The notebook's second-stage seeds (first-stage fits and their regression,
# as it printed them).
ZIEGLER_LN_A = np.array([24.177763, 12.672033, 25.661558, 20.532433])
ZIEGLER_EA_R = np.array([18917.969625, 11703.706227, 20180.373959,
                         16499.900820])  # [K]

# Legacy PharmaPy (branch zachary-dev, Reactors.BatchReactor and
# SimulationExec.EstimateParams) on the same seeds, run 2026-10-27; the
# notebook prints the same values. phi_1 = ln k(T_ref) and phi_2 =
# ln(E/R) with T_ref = 723 K, and the standard deviations from its
# covariance.
LEGACY_ESTIMATES = np.array([-2.021795, -3.331568, -2.152221, -2.301296,
                             9.807026, 9.687626, 9.76952, 9.682907])
LEGACY_STD = np.array([0.07817, 0.21332, 0.1256, 0.07252, 0.05904, 0.15766,
                       0.08987, 0.06618])
LEGACY_OBJECTIVE = 0.2154697923464222

REACTOR_GRID = np.linspace(0.0, 3600.0, 13)  # [s]
CRYST_GRID = np.linspace(0.0, 1200.0, 9)  # [s]


def _synthetic(build, grid, measured, scale):
    """An experiment whose data are the unit's own output, and the unit it
    starts from: its parameters multiplied by ``scale``."""
    truth = build()
    truth.solve_unit(time_grid=grid, verbose=False)
    true_values = Parameters.kinetics(truth).values()

    experiment = Experiment(truth, grid, None, measured)
    data = experiment.measurements(sensitivities=False)

    unit = build()
    parameters = Parameters.kinetics(unit)
    parameters.assign(true_values * scale)

    return Experiment(unit, grid, data, measured), true_values


# ---------------------------------------------------------------------
# No JAX needed
# ---------------------------------------------------------------------

def test_measurements_are_stacked_column_by_column():
    pytest.importorskip("assimulo")

    unit = _reactor(integrator=AssimuloBackend(options={}))
    experiment = Experiment(unit, REACTOR_GRID[1:], np.zeros((12, 2)),
                            measured=('mole_conc_liquid0', ['B', 'C']))
    experiment.parameters = Parameters.kinetics(unit)

    y, sens = experiment.solve(experiment.parameters.values())

    full = unit.result.sensitivities['mole_conc_liquid0'][1:]

    assert y.shape == (12, 2)
    assert sens.shape == (24, 4)
    assert np.array_equal(sens[:12], full[:, 1, :])
    assert np.array_equal(sens[12:], full[:, 2, :])
    assert experiment.state_names == ['B', 'C']


def test_every_experiment_has_to_declare_the_same_parameters():
    reactor = _reactor(integrator=AssimuloBackend(options={}))
    crystallizer = _crystallizer("batch", integrator=AssimuloBackend())

    experiments = [Experiment(reactor, REACTOR_GRID, np.zeros((13, 1)),
                              ('vessel_vol', None)),
                   Experiment(crystallizer, CRYST_GRID, np.zeros((9, 1)),
                              ('vessel_vol', None))]

    with pytest.raises(ValueError, match="same parameters"):
        ParameterEstimation(experiments)


def test_reactor_constants_are_recovered():
    """k_1 and k_2 from 30% off; the activation energies stay fixed."""
    pytest.importorskip("assimulo")

    experiment, truth = _synthetic(
        lambda: _reactor(integrator=AssimuloBackend(options={})),
        REACTOR_GRID, ('mole_conc_liquid0', ['A', 'C', 'D']),
        np.array([1.3, 0.7, 1.0, 1.0]))

    estimation = ParameterEstimation(
        experiment, optimize_flags=[True, True, False, False])
    params, _, _ = estimation.estimate(
        optim_options={'eps_1': 1e-14, 'eps_2': 1e-12}, verbose=False)

    assert params == pytest.approx(truth[:2], rel=1e-4)
    assert estimation.name_params == ['k_1', 'k_2']


# ---------------------------------------------------------------------
# JAX
# ---------------------------------------------------------------------

@pytest.fixture(scope="module")
def jax():
    return pytest.importorskip("jax")


def test_crystal_growth_is_recovered(jax):
    """Growth rate constant and order of the FVM batch crystallizer, from
    the dissolved solute (a solver state) and the third moment (an output
    state)."""
    from PharmaPy.TracedBackends import JaxAssimuloBackend

    scale = np.ones(13)
    scale[[7, 9]] = [1.2, 0.95]  # k_g, g

    experiment, truth = _synthetic(
        lambda: _crystallizer("batch", integrator=JaxAssimuloBackend(
            options={'atol': 1e-9, 'rtol': 1e-9})),
        CRYST_GRID, [('mass_j_liquid0', ['C']), ('mu_n', [3])], scale)

    flags = np.zeros(13, dtype=bool)
    flags[[7, 9]] = True

    # The solute is some 0.4 kg and the third moment many orders of
    # magnitude more, so each column is weighted by its own scale, as
    # measurement variances would.
    weights = np.diag(np.max(np.abs(experiment.y_data), axis=0) ** 2)

    estimation = ParameterEstimation(experiment, optimize_flags=flags,
                                     weight_matrix=weights)
    params, _, _ = estimation.estimate(
        optim_options={'eps_1': 1e-10, 'eps_2': 1e-8}, verbose=False)

    assert params == pytest.approx(truth[[7, 9]], rel=1e-4)


def _ziegler_experiments(integrator):
    with open(os.path.join(ZIEGLER, "ziegler_conc_init.json")) as f:
        initial = json.load(f)

    experiments = []

    for temp in ZIEGLER_TEMPS:
        name = f"{temp}K"
        data = np.genfromtxt(os.path.join(ZIEGLER, f"ziegler_data_{name}.csv"),
                             delimiter=',')

        # Isothermal at the experiment's temperature. Every species has the
        # same molar mass and density here, so the volume (1 L) leaves the
        # concentrations alone; legacy left it unset.
        unit = BatchReactor(integrator=integrator(), isothermal=True)
        unit.Phases = LiquidPhase(ZIEGLER_PATH, temp=temp,
                                  mole_conc=initial[name], vol=1e-3,
                                  name_solv='solvent')
        unit.RxnKinetics = RxnKinetics(
            ZIEGLER_PATH, rxn_list=ZIEGLER_RXNS,
            k_params=np.exp(ZIEGLER_LN_A), ea_params=8.314 * ZIEGLER_EA_R,
            reformulate_kin=True, temp_ref=float(np.mean(ZIEGLER_TEMPS)))

        experiments.append(Experiment(
            unit, data[:, 0], data[:, 1:],
            measured=('mole_conc_liquid0', ['B', 'C']), name=name))

    return experiments


def test_ziegler_matches_legacy_parameter_estimation(jax):
    from PharmaPy.TracedBackends import JaxAssimuloBackend

    estimation = ParameterEstimation(_ziegler_experiments(
        lambda: JaxAssimuloBackend(options={})))
    params, covariance, info = estimation.estimate(
        optim_options={'eps_1': 1e-11, 'eps_2': 1e-11}, verbose=False)

    assert np.all(np.abs(params - LEGACY_ESTIMATES) < 0.01 * LEGACY_STD)
    assert np.sqrt(np.diag(covariance)) == pytest.approx(LEGACY_STD,
                                                          rel=1e-3)
    assert estimation.objfun_iter[-1] == pytest.approx(LEGACY_OBJECTIVE,
                                                       rel=1e-4)

    intervals = estimation.statistics(alpha=0.95).get_intervals()
    assert intervals is not None
