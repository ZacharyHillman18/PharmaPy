"""
Parameter estimation for the refactored units.

Each experiment is a unit of its own, built for that experiment (its
temperature, its initial charge, its feed) and compiled once. Every iterate
of the optimizer assigns the parameters to every experiment's unit, resets
it and solves it with sensitivities; the residuals come from the result and
their Jacobian from ``result.sensitivities``, so no finite differences of
whole solves are taken.

Everything after the model is legacy PharmaPy's own: the Levenberg-Marquardt
optimizer (LevMarq), the covariance, the iteration table, the plots
(ParamEstim.ParameterEstimation) and the confidence intervals
(StatsModule.StatisticsClass). This module only gives them the model
function they expect, func(params, times) -> (y, sens), with the
sensitivities stacked state by state as legacy CVODES runs returned them.

    experiments = [
        Experiment(reactor_at(T), t_data[T], c_data[T],
                   measured=('mole_conc_liquid0', ['B', 'C']), name=f'{T}K')
        for T in temperatures]

    estimation = ParameterEstimation(experiments)
    params, covariance, info = estimation.estimate()
    estimation.plot_data_model()
    estimation.statistics(alpha=0.95).get_intervals()
"""

import time as clock

import numpy as np

from PharmaPy.ParamEstim import ParameterEstimation as LegacyEstimation
from PharmaPy.Parameters import Parameters
from PharmaPy.StatsModule import StatisticsClass


def _as_list(value):
    return list(value) if isinstance(value, (list, tuple)) else [value]


class Experiment:
    """
    One dataset and the unit that models it.

    Parameters
    ----------
    unit : MultiPhaseVessel or ConcurrentVessel
        Built for this experiment only, with an integrator that computes
        sensitivities (AssimuloBackend, JaxAssimuloBackend or
        CasadiBackend). It is reset and solved from its initial state at
        every iterate.
    time_data : array-like
        Sample times [s], starting at or after the unit's start.
    y_data : array-like
        Measured values, (len(time_data), number of measured columns).
    measured : (field, columns) or list of them
        Where each measured column is in the result: a result field
        ('mole_conc_liquid0', 'vessel_vol', ...) and, for a vector field,
        its columns as indices or species names. ``columns=None`` takes the
        whole field. Several pairs measure several fields, in order.
    name : str, optional
        Shown in tables and plots.
    solve_kwargs : dict, optional
        More arguments for solve_unit.
    """

    def __init__(self, unit, time_data, y_data, measured, name=None,
                 solve_kwargs=None):

        self.unit = unit
        self.time_data = np.asarray(time_data, dtype=float).reshape(-1)

        # None for an experiment only simulated (synthetic data, say).
        self.y_data = (None if y_data is None else np.asarray(
            y_data, dtype=float).reshape(self.time_data.size, -1))
        self.name = name
        self.solve_kwargs = dict(solve_kwargs or {})

        if (isinstance(measured, tuple) and len(measured) == 2
                and isinstance(measured[0], str)):
            measured = [measured]

        self.measured = [(field, self._columns(columns))
                         for field, columns in measured]

        self.parameters = None

    def _columns(self, columns):
        if columns is None:
            return None

        names = list(getattr(self.unit, 'name_species', ()) or ())

        return [names.index(column) if isinstance(column, str) else int(column)
                for column in _as_list(columns)]

    @property
    def state_names(self):
        names = list(getattr(self.unit, 'name_species', ()) or ())
        labels = []

        for field, columns in self.measured:
            if columns is None:
                labels.append(field)
            else:
                labels += [names[column] if column < len(names)
                           else f"{field}[{column}]" for column in columns]

        return labels

    def solve(self, values, sensitivities=True):
        """Solve with ``values`` assigned; the measured quantities and, with
        ``sensitivities``, their derivatives in the legacy layout."""

        self.parameters.assign(values)
        self.unit.reset()

        kwargs = dict(verbose=False)
        kwargs.update(self.solve_kwargs)

        self.unit.solve_unit(time_grid=self.time_data,
                             parameters=self.parameters,
                             eval_sens=sensitivities, **kwargs)

        return self.measurements(sensitivities)

    def _rows(self, time):
        """The result rows at the sample times. An event adds a row, so
        the last row at each time is taken."""

        rows = []

        for sample in self.time_data:
            matches = np.flatnonzero(np.isclose(time, sample, rtol=1e-12,
                                                atol=1e-9))

            if matches.size == 0:
                raise RuntimeError(
                    f"{self.name}: the solve reported no point at t="
                    f"{sample:g}.")

            rows.append(matches[-1])

        return rows

    def measurements(self, sensitivities=True):
        """
        y (times, measured columns) and, with ``sensitivities``,
        d(y)/d(free parameters) stacked column by column as legacy
        ParameterEstimation reads them: (columns * times, free).
        """

        result = self.unit.result
        rows = self._rows(np.asarray(result.time, dtype=float))

        values = []
        derivatives = []

        for field, columns in self.measured:
            value = np.asarray(getattr(result, field), dtype=float)[rows]
            value = value.reshape(len(rows), -1)

            if sensitivities:
                sens = np.asarray(result.sensitivities[field], dtype=float)
                sens = sens[rows].reshape(len(rows), value.shape[1], -1)

            if columns is not None:
                value = value[:, columns]

                if sensitivities:
                    sens = sens[:, columns, :]

            values.append(value)

            if sensitivities:
                derivatives.append(sens)

        y = np.hstack(values)

        if not sensitivities:
            return y

        sens = np.concatenate(derivatives, axis=1)

        return y, sens.transpose(1, 0, 2).reshape(-1, sens.shape[-1])


class ParameterEstimation:
    """
    Fit parameters shared by several experiments.

    Parameters
    ----------
    experiments : Experiment, list or dict of them
    parameters : callable, optional
        ``parameters(unit) -> Parameters``, the parameters declared on each
        experiment's unit. Default Parameters.kinetics: every kinetics
        object's concat_params, as legacy estimation fitted. All experiments
        must declare the same parameters, in the same order.
    optimize_flags : list of bool, optional
        Which parameters are fitted; the others stay at their values in the
        first experiment's unit. Default all.
    weight_matrix : array, optional
        As legacy ParameterEstimation: (measured, measured), typically the
        measurement variances on the diagonal.
    sensitivities : bool, optional
        Take the Jacobian from the solves' sensitivities (default). False
        leaves it to legacy finite differences of whole solves, for a
        backend without sensitivities.

    The legacy estimation is ``self.estimation``; its attributes and
    methods (params_convg, covar_params, paramest_df, plot_data_model,
    plot_parity, plot_correlation, ...) are reachable on this object too.
    """

    def __init__(self, experiments, parameters=None, optimize_flags=None,
                 weight_matrix=None, sensitivities=True):

        if isinstance(experiments, Experiment):
            experiments = [experiments]

        if isinstance(experiments, dict):
            for name, experiment in experiments.items():
                experiment.name = experiment.name or name

            experiments = list(experiments.values())

        self.experiments = list(experiments)
        self.sensitivities = bool(sensitivities)

        for index, experiment in enumerate(self.experiments):
            experiment.name = experiment.name or f"exp_{index + 1}"

        declare = Parameters.kinetics if parameters is None else parameters

        for experiment in self.experiments:
            experiment.parameters = declare(experiment.unit)

        first = self.experiments[0].parameters

        for experiment in self.experiments[1:]:
            if experiment.parameters.names != first.names:
                raise ValueError(
                    f"Experiment {experiment.name!r} declares "
                    f"{experiment.parameters.names}, the first "
                    f"{first.names}. Every experiment has to declare the "
                    "same parameters.")

        if optimize_flags is not None:
            for experiment in self.experiments:
                experiment.parameters.free = optimize_flags

        self.parameters = first

        names = [experiment.name for experiment in self.experiments]
        many = len(self.experiments) > 1

        x_data = {name: experiment.time_data
                  for name, experiment in zip(names, self.experiments)}
        y_data = {name: experiment.y_data
                  for name, experiment in zip(names, self.experiments)}
        args = {name: (index,) for index, name in enumerate(names)}

        self.estimation = LegacyEstimation(
            self.model, first.values(),
            x_data if many else self.experiments[0].time_data,
            y_data if many else self.experiments[0].y_data,
            args_fun=args if many else (0,),
            optimize_flags=optimize_flags,
            weight_matrix=weight_matrix,
            name_params=first.names,
            name_states=self.experiments[0].state_names)

        self.seconds = None

    def model(self, params, times, index):
        """The legacy model function: one experiment at ``params``."""

        return self.experiments[index].solve(params, self.sensitivities)

    def estimate(self, optim_options=None, method='LM', verbose=True):
        """
        Run the optimizer; returns (parameters, covariance, info) as legacy
        SimulationExec.EstimateParams did. The fitted values are left
        assigned to every experiment's unit, solved at them.
        """

        tic = clock.perf_counter()

        result = self.estimation.optimize_fn(optim_options=optim_options,
                                             method=method, verbose=verbose)

        self.seconds = clock.perf_counter() - tic

        if verbose:
            print(f"Optimization time: {self.seconds:.2e} s.")

        # The optimizer's last solve may have been a step it rejected.
        fitted = self.estimation.reconstruct_params(
            self.estimation.params_convg)

        for experiment in self.experiments:
            experiment.solve(fitted, self.sensitivities)

        return result

    def statistics(self, alpha=0.95):
        """Confidence intervals and the rest of legacy StatsModule."""

        return StatisticsClass(self.estimation, alpha=alpha)

    def __getattr__(self, name):
        # Only reached for names this object lacks: the legacy estimation's.
        if name == 'estimation':
            raise AttributeError(name)

        return getattr(self.estimation, name)
