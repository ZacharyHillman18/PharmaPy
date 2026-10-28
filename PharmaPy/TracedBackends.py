"""
Integrator backends that compile a unit's right-hand side with JAX.

    JaxBackend           scipy's solve_ivp (BDF by default) on the compiled
                         right-hand side, with its exact Jacobian.
    JaxAssimuloBackend   Assimulo's CVode on the compiled right-hand side,
                         with the exact Jacobian, or exact Jacobian-vector
                         products when the unit asks for a Krylov solver.
    CasadiBackend        CasADi's CVODES on the same compiled function,
                         reached through a casadi.Callback.

All three build the compiled function the same way, in TracedModel: JAX runs
the vessel's own unit_model once with stand-in values and records it. Nothing
about the vessel is rewritten for this; the code it runs through follows the
rules in PharmaPy.xp. What a backend gains:

- a right-hand side that runs as compiled code instead of through the Python
  objects, and
- exact derivatives of it, which replace the difference quotients the other
  backends form column by column (n extra evaluations per Jacobian).

Events and the replay are unchanged. Event functions are evaluated on the unit
itself, numerically, exactly as under ScipyBackend and AssimuloBackend, and
retrieve_results replays the accepted trajectory through the unit's own code.

The compiled function is kept from solve to solve. Numbers declared as
parameters (PharmaPy.Parameters) are inputs of it, so they can change between
solves without a new trace, and sensitivities with respect to them
(eval_sens=True; JaxAssimuloBackend and CasadiBackend) are exact. Any other
number the trace read is frozen into it; before reusing it, each backend
checks it against the unit and traces again, with a warning, if one has
changed (TracedModel.matches).

JAX and CasADi are optional: neither is imported until one of these backends
is constructed.
"""

import warnings
from time import perf_counter

import numpy as np

from PharmaPy import xp
from PharmaPy.IntegratorBackends import (AssimuloBackend, IntegratorBackend,
                                         ScipyBackend)
from PharmaPy._assimulo import Explicit_Problem
from PharmaPy.IntegratorBackends import attach_assimulo_events


def import_jax():
    """JAX, set to double precision, or an error that says how to get it."""

    try:
        import jax
    except ImportError as exc:
        raise ImportError(
            "The traced backends need JAX, which is not installed. "
            "Install it with 'pip install jax' (CPU is enough)."
        ) from exc

    # PharmaPy computes in float64 throughout. JAX defaults to float32, which
    # would silently cost eight digits, so this is not optional. It is a
    # process-wide JAX setting.
    jax.config.update("jax_enable_x64", True)

    return jax


def import_casadi():
    try:
        import casadi
    except ImportError as exc:
        raise ImportError(
            "CasadiBackend needs CasADi, which is not installed. Install it "
            "with 'pip install casadi'."
        ) from exc

    return casadi


# A unit's "krylov" hint is followed from this many states up, and a dense
# factorisation of the exact Jacobian used below it. The hint exists because a
# difference-quotient Jacobian costs n evaluations; the exact one costs about
# as much as one, so the dense solve wins until the factorisation itself
# dominates. Measured on crystallizers: dense won at 9 and 40 states (by 1.6x
# under CVode, 4x under CasADi) and lost at 200 and 205.
KRYLOV_MIN_STATES = 100


def _linear_solver(hint, n_states):
    if hint == "krylov" and n_states < KRYLOV_MIN_STATES:
        return "dense"

    return hint


def _time_argument(time):
    # Always the same type, so the compiled function is never traced again:
    # a Python float and a float64 are different inputs to jax.jit.
    return np.asarray(time, dtype=np.float64)


def _state_argument(states):
    return np.ascontiguousarray(states, dtype=np.float64)


def _parameter_argument(values):
    return np.ascontiguousarray(values, dtype=np.float64).reshape(-1)


class _Traced:
    """
    ``evaluate(copy, time, states)`` compiled as f(time, states, inputs, params).

    Steps 2-4 of TracedModel, for any function of a copy of the unit: its
    right-hand side, or the replay's output states. ``params`` is the
    declared parameters' full vector (empty when none are declared); inside
    the trace it is written to the copy's owners before ``evaluate`` runs,
    so the compiled function reads it rather than the numbers the owners
    held when it was traced.
    """

    def __init__(self, jax, unit, evaluate, time, states, parameters):

        self.traces = 0
        self.time_functions = []

        start = _time_argument(time)
        states = _state_argument(states)
        values = _parameter_argument(
            () if parameters is None else parameters.values())

        def copy_of_unit():
            memo = {}
            pseudo = unit.create_pseudo(memo)
            bound = None if parameters is None else parameters.bind(memo)
            return pseudo, bound

        recorded = []
        discovery, discovery_parameters = copy_of_unit()

        def discover(time, states, params):
            if discovery_parameters is not None:
                discovery_parameters.assign(params)

            return jax.numpy.asarray(evaluate(discovery, time, states))

        with xp.tracing_at(float(time)), xp.time_inputs(record=recorded):
            self.shape = jax.eval_shape(discover, start, states, values).shape

        self.input_shapes = [shape for shapes in recorded for shape in shapes]

        self.pseudo, bound = copy_of_unit()
        pseudo = self.pseudo

        def function(time, states, inputs, params):
            # Runs only while JAX traces. A second trace would run the copy's
            # objects with values the first left behind, so it is refused.
            self.traces += 1

            if self.traces > 1:
                raise RuntimeError(
                    "The right-hand side was traced a second time. Call the "
                    "compiled function with float64 arguments of fixed "
                    "shape, as TracedModel's methods do."
                )

            if bound is not None:
                bound.assign(params)

            with xp.time_inputs(supply=inputs) as supplied:
                out = jax.numpy.asarray(evaluate(pseudo, time, states))

            if supplied.used != len(inputs):
                raise RuntimeError(
                    "The trace met different functions of time the second "
                    "time through, so they cannot be passed in as inputs.")

            self.time_functions = supplied.functions

            return out

        self.compiled = jax.jit(function)

        placeholder = [np.zeros(shape) for shape in self.input_shapes]

        with xp.tracing_at(float(time)):
            self.compiled(start, states, placeholder, values
                          ).block_until_ready()

    def inputs(self, time):
        """The functions of time the trace could not follow, at ``time``."""

        values = []

        for function in self.time_functions:
            leaves, _ = xp._flatten(function(float(time)))
            values += [np.asarray(leaf, dtype=np.float64) for leaf in leaves]

        return values


def _tangent_function(jax, compiled):
    """
    Directional derivatives of compiled f(t, y, inputs, p) along several
    (dy, dp) pairs at once: one jvp per column, mapped, never forming the
    Jacobian. Columns of ``dy`` pair with rows of ``dp``.
    """

    def tangents(time, states, inputs, params, dy, dp):

        def column(state_direction, parameter_direction):
            return jax.jvp(lambda y, p: compiled(time, y, inputs, p),
                           (states, params),
                           (state_direction, parameter_direction))[1]

        return jax.vmap(column, in_axes=(1, 0), out_axes=-1)(dy, dp)

    return jax.jit(tangents)


class TracedModel:
    """
    A unit's right-hand side compiled by JAX, with its exact derivatives.

    Built in steps, each for a reason:

    1. One numeric evaluation of the unit itself. State a controller sets up
       lazily on its first call (DefaultContinuousVesselVolume takes its
       target volume from the first volume it sees) then holds numbers. Were
       the trace the first call, it would hold a traced value.
    2. Copies of the unit (create_pseudo) are what get traced. Tracing leaves
       traced values on the objects it ran through; the unit itself has to
       stay numeric for its events and its replay.
    3. A first, abstract pass notes the functions of time the trace cannot
       follow -- a temperature profile written as plain Python, an upstream
       trajectory -- so the compiled function can take their values as
       inputs (xp.time_inputs). They are evaluated on the number before each
       call, which costs the function itself; calling back into Python from
       the compiled code would cost about 0.2 ms more, every call.
    4. One real trace. The Jacobian and the Jacobian-vector product are
       derived from the compiled function, never by running the Python
       objects again, which would read the traced values left behind.

    Every other number the copy holds is frozen into the compiled function.
    ``parameters`` (PharmaPy.Parameters) names numbers to pass in instead:
    their values are read from the unit's own objects by refresh_parameters,
    which the backends call before each solve, so new values need no new
    trace. ``matches`` tells whether anything else has changed since.

    All methods take and return numpy arrays. ``params``, where a method
    takes it, is a full parameter vector to use instead of the refreshed
    one.
    """

    def __init__(self, unit, time, states, parameters=None):

        jax = import_jax()

        self.jax = jax
        self.n_states = int(np.size(states))
        self.start_time = float(time)

        self.parameters = parameters
        self.parameter_values = _parameter_argument(
            () if parameters is None else parameters.values())

        # How often a non-finite exact derivative was replaced by a
        # difference quotient (see jac and jvp).
        self.patched_columns = 0
        self.patched_products = 0

        # Calls made by the integrator, for comparing backends.
        self.counts = {"rhs": 0, "jac": 0, "jvp": 0, "sens": 0}

        t0 = perf_counter()

        states = _state_argument(states)
        unit.unit_model(time, states)

        self._traced = _Traced(
            jax, unit, lambda pseudo, t, y: pseudo.unit_model(t, y),
            time, states, parameters)

        self._rhs = self._traced.compiled
        self.pseudo = self._traced.pseudo

        self._jac = None
        self._jvp = None
        self._jac_time = None
        self._jac_params = None
        self._sens = None
        self._directions = {}

        self.trace_seconds = perf_counter() - t0
        self.jacobian_seconds = None

    @property
    def traces(self):
        return self._traced.traces

    @property
    def input_shapes(self):
        return self._traced.input_shapes

    @property
    def time_functions(self):
        return self._traced.time_functions

    # ------------------------------------------------------------------
    # Compiled functions
    # ------------------------------------------------------------------

    def inputs(self, time):
        """The functions of time the trace could not follow, at ``time``."""

        return self._traced.inputs(time)

    def refresh_parameters(self):
        """Read the declared parameters' current values from their owners."""

        if self.parameters is not None:
            self.parameter_values = _parameter_argument(
                self.parameters.values())

    def _arguments(self, time, states, params=None):
        return (_time_argument(time), _state_argument(states),
                self.inputs(time),
                self.parameter_values if params is None
                else _parameter_argument(params))

    def rhs(self, time, states, params=None):
        self.counts["rhs"] += 1

        return np.array(self._rhs(*self._arguments(time, states, params)),
                        dtype=float)

    def jac(self, time, states, params=None):
        """Exact d(rhs)/d(states), dense.

        A column whose exact derivative is not finite is replaced by a
        forward difference (see patched_columns). The exact derivative is
        infinite where a fractional power meets zero, such as a kinetic order
        below one at a zero concentration. The difference quotient is the
        finite slope any other backend would have used there.
        """

        if self._jac is None:
            t0 = perf_counter()

            self._jac = self.jax.jit(self.jax.jacfwd(self._rhs, argnums=1))
            self._jac(*self._arguments(time, states, params)
                      ).block_until_ready()

            self.jacobian_seconds = perf_counter() - t0

        self.counts["jac"] += 1

        jacobian = np.array(self._jac(*self._arguments(time, states, params)),
                            dtype=float)

        bad = ~np.isfinite(jacobian).all(axis=0)

        if bad.any():
            states = _state_argument(states)
            base = self.rhs(time, states, params)

            for column in np.flatnonzero(bad):
                step = self._difference_step(states[column])
                shifted = states.copy()
                shifted[column] += step
                jacobian[:, column] = (self.rhs(time, shifted, params)
                                       - base) / step

            self.patched_columns += int(bad.sum())

        return jacobian

    def jvp(self, time, states, vector, params=None):
        """Exact d(rhs)/d(states) times a vector, without forming the matrix.

        Falls back to a directional difference when the exact product is not
        finite, for the reason given in jac.
        """

        if self._jvp is None:
            jax = self.jax
            rhs = self._rhs

            def jvp(time, states, inputs, params, vector):
                return jax.jvp(lambda y: rhs(time, y, inputs, params),
                               (states,), (vector,))[1]

            self._jvp = jax.jit(jvp)

        states = _state_argument(states)
        vector = _state_argument(vector)

        self.counts["jvp"] += 1

        product = np.array(
            self._jvp(*self._arguments(time, states, params), vector),
            dtype=float)

        if not np.isfinite(product).all():
            norm = np.linalg.norm(vector)

            if norm > 0:
                step = self._difference_step(np.linalg.norm(states)) / norm
                product = (self.rhs(time, states + step * vector, params)
                           - self.rhs(time, states, params)) / step

            self.patched_products += 1

        return product

    @staticmethod
    def _difference_step(value):
        return np.sqrt(np.finfo(float).eps) * max(1.0, abs(float(value)))

    def jac_time(self, time, states, params=None):
        """d(rhs)/d(time).

        Exact when everything time-dependent in the model was traced. A
        function of time passed in as an input (see inputs) or run as a host
        callback cannot be differentiated by JAX; this column is then a
        forward difference, at the cost of one more evaluation.
        """

        if self._jac_time is None:
            self._jac_time = False

            if not self.time_functions:
                try:
                    compiled = self.jax.jit(
                        self.jax.jacfwd(self._rhs, argnums=0))
                    compiled(*self._arguments(time, states, params)
                             ).block_until_ready()
                    self._jac_time = compiled
                except Exception:
                    pass

        if self._jac_time:
            return np.array(
                self._jac_time(*self._arguments(time, states, params)),
                dtype=float).reshape(-1, 1)

        step = np.sqrt(np.finfo(float).eps) * max(1.0, abs(float(time)))

        return ((self.rhs(time + step, states, params)
                 - self.rhs(time, states, params)) / step).reshape(-1, 1)

    # ------------------------------------------------------------------
    # Parameters
    # ------------------------------------------------------------------

    def jac_params(self, time, states, params=None):
        """Exact d(rhs)/d(parameters), dense, every declared parameter.

        A non-finite column is replaced by a forward difference, as in jac.
        """

        if self._jac_params is None:
            self._jac_params = self.jax.jit(
                self.jax.jacfwd(self._rhs, argnums=3))

        arguments = self._arguments(time, states, params)
        jacobian = np.array(self._jac_params(*arguments), dtype=float)

        bad = ~np.isfinite(jacobian).all(axis=0)

        if bad.any():
            values = arguments[3].copy()
            base = self.rhs(time, states, values)

            for column in np.flatnonzero(bad):
                step = self._difference_step(values[column])
                shifted = values.copy()
                shifted[column] += step
                jacobian[:, column] = (self.rhs(time, states, shifted)
                                       - base) / step

        return jacobian

    def _parameter_directions(self, free_index):
        key = tuple(int(index) for index in free_index)

        if key not in self._directions:
            self._directions[key] = np.ascontiguousarray(
                np.eye(self.parameter_values.size)[list(key)])

        return self._directions[key]

    def tangents(self, time, states, state_directions, parameter_directions,
                 params=None):
        """
        d(rhs)/d(states) @ state_directions
        + d(rhs)/d(params) @ parameter_directions.T, exact.

        ``state_directions`` is (states, k) and ``parameter_directions``
        (k, every declared parameter): column j of the result is the
        derivative along the pair (state_directions[:, j],
        parameter_directions[j]). One jvp per column, so neither Jacobian
        is formed, which is what lets a Krylov-solved unit carry
        sensitivities.
        """

        if self._sens is None:
            self._sens = _tangent_function(self.jax, self._rhs)

        state_directions = np.ascontiguousarray(
            np.reshape(state_directions, (self.n_states, -1)),
            dtype=np.float64)
        parameter_directions = np.ascontiguousarray(
            np.reshape(parameter_directions,
                       (state_directions.shape[1], -1)), dtype=np.float64)
        arguments = self._arguments(time, states, params)

        self.counts["sens"] += 1

        out = np.array(self._sens(*arguments, state_directions,
                                  parameter_directions), dtype=float)

        if not np.isfinite(out).all():
            # Rare (see jac): the patched Jacobians instead.
            out = (self.jac(time, states, arguments[3]) @ state_directions
                   + self.jac_params(time, states, arguments[3])
                   @ parameter_directions.T)

        return out

    def sens_rhs(self, time, states, sens, free_index, params=None):
        """
        The forward sensitivity equations' right-hand side,
        d(rhs)/d(states) @ sens + d(rhs)/d(params)[:, free_index], exact.

        ``sens`` is (states, free parameters); see tangents.
        """

        return self.tangents(time, states, sens,
                             self._parameter_directions(free_index), params)

    # ------------------------------------------------------------------
    # Still the unit's model?
    # ------------------------------------------------------------------

    def matches(self, unit, time, states, rtol=1e-9):
        """
        Whether the compiled right-hand side still computes what the unit
        does, at ``states`` and at a point nudged from it.

        Everything the trace read that is not a declared parameter is frozen
        into the compiled function: change a kinetic constant with
        set_params, a controller's set point, a feed or the utility after
        compiling, and the compiled function would go on with the old value
        while the unit's own code uses the new one. This finds that out by
        evaluating both. The two agree to about 1e-14 when nothing changed.

        The nudged point (every state moved by up to 0.1%) catches a setting
        that does not show at ``states`` itself, such as a controller gain
        while its error is zero. ``states`` is evaluated last, so the unit is
        left where it was.
        """

        states = _state_argument(states)
        rng = np.random.default_rng(0)
        nudged = states * (1 + 1e-3 * rng.uniform(-1, 1, states.size))

        agree = True

        for point in (nudged, states):
            numeric = np.asarray(unit.unit_model(time, point), dtype=float)
            compiled = np.asarray(self._rhs(*self._arguments(time, point)),
                                  dtype=float)

            finite = np.isfinite(numeric)
            scale = np.max(np.abs(numeric[finite]), initial=0.0)

            close = ((np.abs(compiled - numeric)
                      <= rtol * np.abs(numeric) + 1e-12 * scale)
                     | (compiled == numeric)
                     | (np.isnan(compiled) & np.isnan(numeric)))

            agree = agree and bool(close.all())

        return agree

    # ------------------------------------------------------------------
    # CasADi
    # ------------------------------------------------------------------

    def casadi_function(self, name="pharmapy_rhs", free_index=None):
        """
        The compiled right-hand side as a casadi.Function f(t, x) -> dx/dt.

        Its derivatives are the exact ones above, so an integrator, a
        collocation transcription or an NLP built on it gets exact Jacobians
        from JAX. The returned function keeps the objects behind it alive.

        ``free_index`` makes it f(t, x, p), p being those entries of the
        declared parameters (the others keep their refreshed values).
        """

        return _casadi_rhs(self, name, free_index)


class TracedOutputs:
    """
    The replay's output states compiled as h(time, states, params), and how
    they move along sensitivity directions.

    What is measured is mostly an output state -- a concentration, the
    volume, a supersaturation -- computed from the solver states by the
    replay (MultiPhaseVessel.output_states_at). Its sensitivity to a
    parameter is dh/dy S + dh/dp, which this gives exactly, compiled the way
    TracedModel compiles the right-hand side.
    """

    def __init__(self, unit, time, states, parameters=None):

        jax = import_jax()

        self.jax = jax

        states = _state_argument(states)
        sample = unit.create_pseudo().output_states_at(
            float(time), unit.solver_state_collection.unpack(states))

        # Where each output state sits in the flat vector h returns.
        self.layout = []
        start = 0

        for key, value in sample.items():
            shape = np.shape(value)
            size = int(np.prod(shape, dtype=int))
            self.layout.append((key, shape, slice(start, start + size)))
            start += size

        keys = [key for key, _, _ in self.layout]

        def evaluate(pseudo, time, states):
            values = pseudo.output_states_at(
                time, pseudo.solver_state_collection.unpack(states))

            return xp.concatenate([xp.ravel(xp.asarray(values[key],
                                                       dtype=float))
                                   for key in keys])

        self._traced = _Traced(jax, unit, evaluate, time, states, parameters)
        self._tangents = None

    @property
    def traces(self):
        return self._traced.traces

    def tangents(self, time, states, sens, free_index, params):
        """{StateKey: (*shape, free)}: dh/dy @ sens + dh/dp[:, free_index]."""

        if self._tangents is None:
            self._tangents = _tangent_function(self.jax,
                                               self._traced.compiled)

        params = _parameter_argument(params)
        directions = np.ascontiguousarray(
            np.eye(params.size)[list(free_index)])

        out = np.array(self._tangents(
            _time_argument(time), _state_argument(states),
            self._traced.inputs(time), params,
            np.ascontiguousarray(sens, dtype=np.float64), directions),
            dtype=float)

        columns = out.shape[-1]

        return {key: out[part].reshape(shape + (columns,))
                for key, shape, part in self.layout}


def _casadi_rhs(model, name, free_index=None):
    """casadi.Callback pair: f(t, x) and its Jacobian, evaluated by JAX.

    With ``free_index`` (positions in the model's parameter vector) the
    function is f(t, x, p) instead, p being those parameters, with exact
    derivatives in p as well. That is what CasADi's forward sensitivities
    differentiate.
    """

    ca = import_casadi()
    n = model.n_states

    with_p = free_index is not None
    free = [] if free_index is None else [int(i) for i in free_index]
    k = len(free)

    # Input sizes, in order: t, x and, with parameters, p.
    sizes = (1, n, k) if with_p else (1, n)
    n_inputs = len(sizes)

    def full(arg):
        """The model's parameter vector with the free entries from p."""
        values = model.parameter_values.copy()

        if with_p:
            values[free] = np.ravel(arg[2].full())

        return values

    directions = np.eye(model.parameter_values.size)[free]

    class Jacobian(ca.Callback):

        def __init__(self, name, opts):
            ca.Callback.__init__(self)
            self.construct(name, opts)

        # Inputs: the function's, and the nominal output CasADi passes along.
        def get_n_in(self):
            return n_inputs + 1

        # d(f)/d(each input).
        def get_n_out(self):
            return n_inputs

        def get_sparsity_in(self, index):
            return ca.Sparsity.dense((sizes + (n,))[index], 1)

        def get_sparsity_out(self, index):
            return ca.Sparsity.dense(n, sizes[index])

        def eval(self, arg):
            time = float(arg[0])
            states = np.ravel(arg[1].full())
            params = full(arg)

            out = [model.jac_time(time, states, params),
                   model.jac(time, states, params)]

            if with_p:
                out.append(model.jac_params(time, states, params)[:, free])

            return out

    class Forward(ca.Callback):
        """Directional derivatives.

        A few directions -- one per Krylov iteration, which is what CasADi's
        GMRES Newton scheme asks for -- are one JAX jvp each, far cheaper
        than forming the Jacobian. CasADi also builds the Jacobian for its
        direct solve out of forward directions, all n of them at once; that
        request is the exact Jacobian times the seeds. Forward sensitivities
        ask for one direction per parameter, seeded in x and p together.

        CasADi's forward sensitivities also differentiate this once more,
        for the Newton matrix of the system they add. Only the iteration
        uses that, not the converged values, so CasADi differences it
        (enable_fd) rather than this building second derivatives.
        """

        def __init__(self, n_directions, name, opts):
            ca.Callback.__init__(self)
            self.n_directions = n_directions
            self.construct(name, dict(opts, enable_fd=with_p))

        # Inputs: the function's, the nominal output, then a seed for each
        # of the function's inputs.
        def get_n_in(self):
            return 2 * n_inputs + 1

        def get_n_out(self):
            return 1

        def get_sparsity_in(self, index):
            if index < n_inputs:
                return ca.Sparsity.dense(sizes[index], 1)

            if index == n_inputs:
                return ca.Sparsity.dense(n, 1)

            return ca.Sparsity.dense(sizes[index - n_inputs - 1],
                                     self.n_directions)

        def get_sparsity_out(self, index):
            return ca.Sparsity.dense(n, self.n_directions)

        def eval(self, arg):
            time = float(arg[0])
            states = np.ravel(arg[1].full())
            params = full(arg)

            seeds = arg[n_inputs + 1:]
            seed_time = np.ravel(seeds[0].full())
            seed_states = np.asarray(seeds[1].full()).reshape(
                n, self.n_directions)
            seed_params = (np.asarray(seeds[2].full()).reshape(
                k, self.n_directions) if with_p else None)

            by_time = (model.jac_time(time, states, params).ravel()
                       if np.any(seed_time) else None)

            if seed_params is not None and np.any(seed_params):
                out = model.tangents(time, states, seed_states,
                                     seed_params.T @ directions, params)

            elif self.n_directions > max(1, n // 16):
                out = model.jac(time, states, params) @ seed_states

            else:
                out = np.zeros((n, self.n_directions))

                for column in range(self.n_directions):

                    if np.any(seed_states[:, column]):
                        out[:, column] = model.jvp(
                            time, states, seed_states[:, column], params)

            if by_time is not None:
                out = out + np.outer(by_time, seed_time)

            return [out]

    class RHS(ca.Callback):

        def __init__(self, name, opts):
            ca.Callback.__init__(self)
            self.construct(name, opts)

        def get_n_in(self):
            return n_inputs

        def get_n_out(self):
            return 1

        def get_name_in(self, index):
            return ("t", "x", "p")[index]

        def get_name_out(self, index):
            return "ode"

        def get_sparsity_in(self, index):
            return ca.Sparsity.dense(sizes[index], 1)

        def get_sparsity_out(self, index):
            return ca.Sparsity.dense(n, 1)

        def eval(self, arg):
            return [model.rhs(float(arg[0]), np.ravel(arg[1].full()),
                              full(arg))]

        def has_jacobian(self):
            return True

        def get_jacobian(self, name, inames, onames, opts):
            # CasADi keeps no reference to the Callback objects it gets back,
            # so this one has to, every one of them: it may ask for several.
            return self._keep(name, Jacobian(name, opts))

        def has_forward(self, n_directions):
            return True

        def get_forward(self, n_directions, name, inames, onames, opts):
            return self._keep(name, Forward(n_directions, name, opts))

        def _keep(self, name, callback):
            self.__dict__.setdefault("_derivatives", {})[name] = callback
            return callback

    return RHS(name, {})


def _casadi_dae(model, algebraic, name="pharmapy_dae"):
    """casadi.Callback pair for a semi-explicit DAE: (ode, alg)(t, x, z).

    The unit's state vector holds differential and algebraic entries mixed,
    in the order its StateCollection lays them out, and unit_model returns a
    rate for each differential entry and a residual for each algebraic one.
    CasADi's IDAS wants them apart; ``algebraic`` (a boolean mask over the
    state vector) says which is which.
    """

    ca = import_casadi()

    algebraic = np.asarray(algebraic, dtype=bool)
    differential = ~algebraic
    nx = int(differential.sum())
    nz = int(algebraic.sum())
    n = algebraic.size

    def assemble(x, z):
        states = np.empty(n)
        states[differential] = np.ravel(x.full())
        states[algebraic] = np.ravel(z.full())
        return states

    class Jacobian(ca.Callback):

        def __init__(self, name, opts):
            ca.Callback.__init__(self)
            self.construct(name, opts)

        # Inputs: t, x, z, and the nominal ode and alg outputs.
        def get_n_in(self):
            return 5

        # d(ode)/d(t, x, z), then d(alg)/d(t, x, z).
        def get_n_out(self):
            return 6

        def get_sparsity_in(self, index):
            return ca.Sparsity.dense((1, nx, nz, nx, nz)[index], 1)

        def get_sparsity_out(self, index):
            rows = nx if index < 3 else nz
            return ca.Sparsity.dense(rows, (1, nx, nz)[index % 3])

        def eval(self, arg):
            time = float(arg[0])
            states = assemble(arg[1], arg[2])

            full = model.jac(time, states)
            by_time = model.jac_time(time, states)

            out = []

            for rows in (differential, algebraic):
                out += [by_time[rows], full[np.ix_(rows, differential)],
                        full[np.ix_(rows, algebraic)]]

            return out

    class DAE(ca.Callback):

        def __init__(self, name, opts):
            ca.Callback.__init__(self)
            self.construct(name, opts)

        def get_n_in(self):
            return 3

        def get_n_out(self):
            return 2

        def get_name_in(self, index):
            return ("t", "x", "z")[index]

        def get_name_out(self, index):
            return ("ode", "alg")[index]

        def get_sparsity_in(self, index):
            return ca.Sparsity.dense((1, nx, nz)[index], 1)

        def get_sparsity_out(self, index):
            return ca.Sparsity.dense((nx, nz)[index], 1)

        def eval(self, arg):
            rates = model.rhs(float(arg[0]), assemble(arg[1], arg[2]))
            return [rates[differential], rates[algebraic]]

        def has_jacobian(self):
            return True

        def get_jacobian(self, name, inames, onames, opts):
            self._jacobian = getattr(self, "_jacobian", {})
            self._jacobian[name] = Jacobian(name, opts)
            return self._jacobian[name]

    return DAE(name, {})


def _casadi_events(unit, n_states):
    """The unit's event functions as a casadi.Callback g(t, x).

    Evaluated on the unit itself, numerically, like every other backend
    does. CasADi only needs their values to locate a crossing; enable_fd
    covers it should it ask for a derivative.
    """

    ca = import_casadi()
    n_events = len(unit.compiled_events)

    class Events(ca.Callback):

        def __init__(self):
            ca.Callback.__init__(self)
            self.construct("pharmapy_events", {"enable_fd": True})

        def get_n_in(self):
            return 2

        def get_n_out(self):
            return 1

        def get_sparsity_in(self, index):
            return ca.Sparsity.dense((1, n_states)[index], 1)

        def get_sparsity_out(self, index):
            return ca.Sparsity.dense(n_events, 1)

        def eval(self, arg):
            values = unit.evaluate_events(float(arg[0]),
                                          np.ravel(arg[1].full()))
            return [np.asarray(values, dtype=float)]

    return Events()


def _casadi_identity_transition(n_states):
    """Leave the state unchanged at an event.

    Every event the backend accepts only marks a regime change, and CasADi
    has to be told so: without a transition function its default does not
    leave the state where it was.
    """

    ca = import_casadi()

    index = ca.MX.sym("index")
    t = ca.MX.sym("t")
    x = ca.MX.sym("x", n_states)
    z = ca.MX.sym("z", 0)
    p = ca.MX.sym("p", 0)
    u = ca.MX.sym("u", 0)

    return ca.Function("pharmapy_transition", [index, t, x, z, p, u], [x, z],
                       ca.event_in(), ca.event_out())


# ======================================================================
# Backends
# ======================================================================

class _CompiledModelCheck:
    """
    What every traced backend does before reusing its compiled model.

    The compiled model is reused from solve to solve, which is what makes a
    loop of short solves cheap. Two things keep it right:

    - the declared parameters' values are read from their owners again, so
      assigning parameters between solves needs no new trace, and
    - the compiled right-hand side is checked against the unit's own
      (TracedModel.matches). A number the trace froze in that has changed
      since -- a kinetic constant changed with set_params without being
      declared, a controller setting, a feed, the utility -- would otherwise
      be silently ignored. When the check fails the model is traced again
      at the unit's current state, with a warning; the unit is not reset.
    """

    parameters_are_inputs = True

    def solve(self, unit, runtime=None, time_grid=None, eval_sens=False,
              jac_v_prod=False, verbose=True, options=None, any_event=True,
              parameters=None):

        if not self.needs_compile(unit, eval_sens, jac_v_prod, parameters):
            self.check_compiled_model(unit)

        return super().solve(
            unit, runtime=runtime, time_grid=time_grid, eval_sens=eval_sens,
            jac_v_prod=jac_v_prod, verbose=verbose, options=options,
            any_event=any_event, parameters=parameters)

    def check_compiled_model(self, unit):

        self.model.refresh_parameters()

        time = float(unit.elapsed_time)
        states = np.asarray(unit.create_solver_init_states(), dtype=float)

        if self.model.matches(unit, time, states):
            return

        warnings.warn(
            f"{type(unit).__name__}: a number its compiled model was built "
            "with has changed -- a kinetic constant set with set_params, a "
            "controller setting, a feed or the utility -- so the model is "
            f"traced again at t={time:g}. Numbers changed between solves "
            "need no new trace when they are declared: solve_unit("
            "parameters=Parameters.kinetics(unit)), or Parameters().add(...) "
            "for anything else.",
            RuntimeWarning,
            stacklevel=4,
        )

        self.retrace(unit, time, states)

    def retrace(self, unit, time, states):
        self.model = TracedModel(unit, time, states, self.parameters)
        self.outputs = None

    def output_tangents(self, unit):
        """The output states' sensitivities, from the compiled replay
        (TracedOutputs), built at the first point it is asked about."""

        backend = self

        def tangents(time, states, sens, params):

            if backend.outputs is None:
                backend.outputs = TracedOutputs(unit, time, states,
                                                backend.parameters)

            return backend.outputs.tangents(
                time, states, sens, backend.parameters.free_index, params)

        return tangents


class JaxBackend(_CompiledModelCheck, ScipyBackend):
    """
    ScipyBackend on the JAX-compiled right-hand side.

    Everything ScipyBackend does is kept: the methods, the option names, the
    event segmentation, the stall detector. Two things change: the right-hand
    side solve_ivp calls is the compiled one, and the implicit methods (BDF,
    Radau, LSODA) are handed its exact Jacobian instead of building one from
    n difference quotients.

    Parameters
    ----------
    exact_jacobian : bool, optional
        Pass the exact Jacobian to solve_ivp. Default True. False keeps the
        compiled right-hand side and lets scipy difference it, which isolates
        how much of the speed-up comes from each.

    The other parameters are ScipyBackend's.
    """

    IMPLICIT_METHODS = ("BDF", "Radau", "LSODA")

    def __init__(self, options=None, method="BDF", exact_jacobian=True,
                 **kwargs):

        import_jax()

        super().__init__(options=options, method=method, **kwargs)

        self.exact_jacobian = bool(exact_jacobian)
        self.model = None
        self.outputs = None

    def compile_integrator(self, unit, eval_sens=False, jac_v_prod=False,
                           options=None, verbose=True, any_event=True,
                           parameters=None):

        states_init = super().compile_integrator(
            unit, eval_sens, jac_v_prod, options, verbose, any_event,
            parameters)

        self.model = TracedModel(unit, unit.elapsed_time, states_init,
                                 self.parameters)

        return states_init

    def evaluate_rhs(self, unit, time, states):
        return self.model.rhs(time, states)

    def translate_options(self):

        translated = super().translate_options()

        if self.exact_jacobian and self.method in self.IMPLICIT_METHODS:
            translated.setdefault("jac", self.model.jac)

        return translated


class JaxAssimuloBackend(_CompiledModelCheck, AssimuloBackend):
    """
    AssimuloBackend (CVode) on the JAX-compiled right-hand side.

    CVode gets the compiled right-hand side and exact derivatives of it, in
    the form its linear solver uses: the dense Jacobian under DENSE, and
    Jacobian-vector products under SPGMR. The difference quotients CVode
    would otherwise form -- n evaluations per Jacobian, or one per Krylov
    iteration -- are not needed. A crystallizer asks for SPGMR
    (linear_solver_hint = "krylov"); that is followed from KRYLOV_MIN_STATES
    states up, and below that the exact dense Jacobian is the faster choice.

    Sensitivities (eval_sens=True) are CVODES's forward sensitivities with
    an exact right-hand side for them (TracedModel.sens_rhs), and the output
    states' sensitivities come from the compiled replay (TracedOutputs).

    Events work exactly as under AssimuloBackend: they are evaluated on the
    unit itself.

    Parameters
    ----------
    exact_jacobian : bool, optional
        Hand CVode the exact derivatives. Default True. False keeps only the
        compiled right-hand side, and leaves the sensitivities to CVODES's
        difference quotients.
    """

    def __init__(self, options=None, exact_jacobian=True):

        import_jax()

        # A fresh dict per instance; AssimuloBackend's own default is one
        # dict shared by every instance. Only None takes the default: an
        # empty dict means no options, as it does for AssimuloBackend (it
        # used to fall to maxh=1, which on a six-hour CSTR is 21,600 steps).
        super().__init__(options={'maxh': 1} if options is None
                         else dict(options))

        self.exact_jacobian = bool(exact_jacobian)
        self.model = None
        self.outputs = None

    def compile_integrator(self, unit, eval_sens=False, jac_v_prod=False,
                           options=None, verbose=True, any_event=True,
                           parameters=None):

        self.outputs = None

        states_init = super().compile_integrator(
            unit, eval_sens, jac_v_prod, options, verbose, any_event,
            parameters)

        if self.exact_jacobian:
            self._solver.usejac = True

        return states_init

    def set_linear_solver(self, kind):
        """The unit's request, with Krylov kept for large systems only.

        With exact derivatives a small system is solved faster densely; see
        KRYLOV_MIN_STATES.
        """

        if self.exact_jacobian and self.model is not None:
            kind = _linear_solver(kind, self.model.n_states)

        super().set_linear_solver(kind)

    def set_ode_problem(self, unit, states_init):

        self.model = TracedModel(unit, unit.elapsed_time, states_init,
                                 self.parameters)

        # The callbacks look the model up on the backend at each call, so a
        # model traced again (check_compiled_model) takes over in place.
        backend = self
        events = unit.compiled_events
        settings = {}

        if events:
            settings["sw0"] = [True] * len(events)

        if self.eval_sens:
            # CVode passes the parameters on: positionally as (t, y, p)
            # without switches, by keyword (sw=, p=) with them; jacv always
            # by keyword. They are the free entries of the full vector.
            free = self.parameters.free_index
            settings["p0"] = self.parameters.free_values()

            def full(p):
                values = backend.model.parameter_values.copy()

                if p is not None:
                    values[free] = p

                return values

            def rhs(time, states, p=None, sw=None):
                return backend.model.rhs(time, states, full(p))

            def jac(time, states, p=None, sw=None):
                return backend.model.jac(time, states, full(p))

            def jacv(time, states, rhs_value, vector, p=None, sw=None):
                return backend.model.jvp(time, states, vector, full(p))

            def rhs_sens(time, states, sens, p=None, sw=None):
                return backend.model.sens_rhs(time, states, sens, free,
                                              full(p))

        else:
            def rhs(time, states, sw=None):
                return backend.model.rhs(time, states)

            def jac(time, states, sw=None):
                return backend.model.jac(time, states)

            def jacv(time, states, rhs_value, vector, sw=None):
                return backend.model.jvp(time, states, vector)

        problem = Explicit_Problem(rhs, states_init, t0=unit.elapsed_time,
                                   **settings)

        if events:
            attach_assimulo_events(problem, unit,
                                   carry_sensitivities=self.eval_sens)

        if self.exact_jacobian:
            problem.jac = jac
            problem.jacv = jacv

            if self.eval_sens:
                problem.rhs_sens = rhs_sens

        self._problem = problem


class CasadiBackend(_CompiledModelCheck, IntegratorBackend):
    """
    CasADi's CVODES (or IDAS) on the JAX-compiled right-hand side.

    The model reaches CasADi as a casadi.Callback that evaluates the compiled
    function and gives CasADi its exact Jacobian (TracedModel.casadi_function,
    which is also the entry point for building an MPC or estimation problem
    around a unit). No part of PharmaPy is written in CasADi's own symbols.

    A unit with algebraic states -- a ConcurrentVessel whose recycle is
    carried as algebraic tear streams (tear_mode='algebraic') -- goes to IDAS
    as a semi-explicit DAE, which also computes consistent initial algebraic
    values before the first step.

    Results are reported on a time grid: ``time_grid`` when given, otherwise
    ``n_points`` evenly spaced times over the run.

    The unit's events (supersaturation crossing zero, an outlet flow changing
    sign) only mark where the right-hand side has a corner, and by default
    CVODES steps through them under its error control. ``locate_events=True``
    hands them to CasADi's zero-crossing detection instead, which stops and
    restarts at each as CVode does under AssimuloBackend. That is costly
    here: CasADi then wants the forward derivative of the model at every
    output time, and of the event functions, which are numeric and get
    differenced. CasADi 3.8 also marks the support experimental.

    Sensitivities (eval_sens=True) are CasADi's: the free parameters become
    the DAE's ``p``, and CasADi's forward sensitivities of the integrator
    run on the Callback's exact derivatives in x and p.

    What it does not do, and refuses rather than fakes:

    - **Terminal events.** CasADi restarts at a crossing but this backend has
      no way yet to end the run there, so a unit with an event that ends the
      run is refused.
    - **Sensitivities with algebraic states or located events**, for now.

    Parameters
    ----------
    options : dict, optional
        Integrator options, in Assimulo's names (atol, rtol, maxh, maxsteps)
        or CasADi's (abstol, reltol, max_step_size, max_num_steps, ...).
        Tolerances default to 1e-6, as CVode's do.
    n_points : int, optional
        Output times when no time_grid is given. Default 201.
    locate_events : bool, optional
        Stop and restart at the unit's events. Default False.
    """

    supports_algebraic = True

    OPTION_ALIASES = {
        "atol": "abstol",
        "rtol": "reltol",
        "maxh": "max_step_size",
        "maxsteps": "max_num_steps",
        "inith": "step0",
        "maxord": "max_multistep_order",
    }

    IGNORED_OPTIONS = frozenset({
        "verbosity", "report_continuously", "time_limit", "clock_step",
        "iter", "discr", "usejac",
    })

    # Assimulo's linear solver names. Anything else given as linear_solver
    # is taken to be one of CasADi's and passed through.
    ASSIMULO_LINEAR_SOLVERS = {
        "DENSE": {},
        "SPARSE": {},
        "SPGMR": {"newton_scheme": "gmres"},
    }

    def __init__(self, options=None, n_points=201, locate_events=False):

        import_jax()
        import_casadi()

        super().__init__()

        self.options = dict(options) if options else {}
        self.n_points = int(n_points)
        self.locate_events = bool(locate_events)
        self.model = None
        self.function = None
        self._stats = {}

    def translate_options(self, algebraic=False):

        translated = {}

        for name, value in self.options.items():

            if name in self.IGNORED_OPTIONS:
                continue

            if (name == "linear_solver"
                    and value in self.ASSIMULO_LINEAR_SOLVERS):
                translated.update(self.ASSIMULO_LINEAR_SOLVERS[value])
                continue

            translated[self.OPTION_ALIASES.get(name, name)] = value

        translated.setdefault("abstol", 1e-6)
        translated.setdefault("reltol", 1e-6)

        # CSparse's LU, not CasADi's default sparse QR. On the moments
        # crystallizer, whose states span some thirty orders of magnitude,
        # the QR lost the Newton iteration: 8438 steps and 6355 convergence
        # failures to reach t = 1.4 s, against 205 and 2 with this.
        translated.setdefault("linear_solver", "csparse")

        # A large unit's "krylov" hint becomes CasADi's GMRES Newton scheme
        # on exact Jacobian-vector products (the Callback's forward
        # derivatives), unpreconditioned as CVode's SPGMR is under
        # AssimuloBackend. That avoids CasADi's habit of forming the dense
        # Jacobian at every linear setup: on a 205-state crystallizer, 10-17 s
        # instead of 50. A small unit is solved densely (KRYLOV_MIN_STATES).
        # The DAE path (IDAS) stays direct.
        if (not algebraic and self.model is not None
                and _linear_solver(self.linear_solver,
                                   self.model.n_states) == "krylov"):
            translated.setdefault("newton_scheme", "gmres")
            translated.setdefault("use_preconditioner", False)

        return translated

    def compile_integrator(self, unit, eval_sens=False, jac_v_prod=False,
                           options=None, verbose=True, any_event=True,
                           parameters=None):

        if eval_sens and (unit.has_algebraic_balance or self.locate_events):
            raise NotImplementedError(
                "CasadiBackend computes sensitivities for ODEs with events "
                "stepped through (locate_events=False) only. Use "
                "JaxAssimuloBackend."
            )

        self.check_algebraic_support(unit)

        terminal = [event.name for event in unit.compiled_events
                    if getattr(event, "terminal", False)]

        if terminal:
            raise NotImplementedError(
                f"CasadiBackend cannot stop at an event, and {terminal} "
                "end the run. Use JaxBackend or JaxAssimuloBackend, which "
                "locate them."
            )

        if options:
            self.options.update(options)

        unit.reset()

        states_init = np.asarray(unit.create_solver_init_states(),
                                 dtype=float)
        unit.save_initial_solver_state(states_init, unit.elapsed_time)

        self.eval_sens = eval_sens
        self.jac_v_prod = jac_v_prod
        self.set_parameters(unit, parameters, eval_sens)

        self.model = TracedModel(unit, unit.elapsed_time, states_init,
                                 self.parameters)
        self.outputs = None
        self.function = self._casadi_function()

        unit.configure_solver()

        self._compiled = True

        return states_init

    def _casadi_function(self):
        """f(t, x), or f(t, x, p) over the free parameters for sensitivities."""

        if self.eval_sens:
            return self.model.casadi_function(
                free_index=self.parameters.free_index)

        return self.model.casadi_function()

    def solve(self, unit, runtime=None, time_grid=None, eval_sens=False,
              jac_v_prod=False, verbose=True, options=None, any_event=True,
              parameters=None):

        # The check _CompiledModelCheck.solve would make; this solve is
        # CasADi's own, so it makes it here.
        if self.needs_compile(unit, eval_sens, jac_v_prod, parameters):
            self.compile_integrator(unit, eval_sens, jac_v_prod, options,
                                    verbose, any_event, parameters)
        else:
            self.check_compiled_model(unit)

        unit.derivatives = np.array(
            unit.unit_model(time=unit.elapsed_time,
                            states=unit.create_solver_init_states()),
            dtype=float)

        time, states = self.fast_solve(unit, runtime=runtime,
                                       time_grid=time_grid, verbose=verbose)

        unit.retrieve_results(time, states)

        if self.eval_sens:
            unit.retrieve_sensitivities(time, states, self._sensitivities,
                                        self.parameters,
                                        self.output_tangents(unit))

        return time, states

    def retrace(self, unit, time, states):

        super().retrace(unit, time, states)
        self.function = self._casadi_function()

    def fast_solve(self, unit, runtime=None, time_grid=None, verbose=True):

        if not self._compiled:
            raise RuntimeError("Integrator has not been compiled.")

        ca = import_casadi()

        states_init = np.asarray(unit.create_solver_init_states(),
                                 dtype=float)
        start_time = float(unit.elapsed_time)

        if time_grid is not None:
            grid = np.asarray(time_grid, dtype=float).reshape(-1)
            grid = grid[grid > start_time]
        elif runtime is not None:
            grid = np.linspace(start_time, start_time + float(runtime),
                               self.n_points)[1:]
        else:
            raise ValueError("Either runtime or time_grid must be supplied.")

        opts = self.translate_options(algebraic=unit.has_algebraic_balance)

        if not verbose:
            opts.setdefault("print_stats", False)

        if unit.has_algebraic_balance:
            return self._solve_dae(unit, states_init, start_time, grid, opts)

        if self.eval_sens:
            return self._solve_with_sensitivities(
                states_init, start_time, grid, opts)

        t = ca.MX.sym("t")
        x = ca.MX.sym("x", states_init.size)

        dae = {"t": t, "x": x, "ode": self.function(t, x)}

        if self.locate_events and unit.compiled_events:
            # The vessel's switching surfaces, located by CasADi and
            # restarted from, as CVode does under AssimuloBackend.
            self._events = _casadi_events(unit, states_init.size)
            dae["zero"] = self._events(t, x)
            opts["transition"] = _casadi_identity_transition(
                states_init.size)

        integrator = ca.integrator("pharmapy", "cvodes", dae,
                                   start_time, list(grid), opts)

        result = integrator(x0=states_init)
        self._stats = integrator.stats()

        states = np.vstack((states_init[None, :],
                            np.asarray(result["xf"].full()).T))
        times = np.concatenate(([start_time], grid))

        return times, states

    def _solve_with_sensitivities(self, states_init, start_time, grid, opts):
        """
        CVODES with the free parameters as p, and d(x)/d(p) at every output
        time from CasADi's forward sensitivities (one direction per free
        parameter, run on the Callback's exact forward derivatives).
        """

        ca = import_casadi()

        free = self.parameters.free_index
        n = states_init.size

        t = ca.MX.sym("t")
        x = ca.MX.sym("x", n)
        p = ca.MX.sym("p", free.size)

        integrator = ca.integrator(
            "pharmapy", "cvodes",
            {"t": t, "x": x, "p": p, "ode": self.function(t, x, p)},
            start_time, list(grid), opts)

        with_sensitivities = integrator.factory(
            "pharmapy_sens", ["x0", "p"], ["xf", "jac:xf:p"])

        values = self.model.parameter_values[free]
        xf, jacobian = with_sensitivities(states_init, values)
        self._stats = with_sensitivities.stats()

        states = np.vstack((states_init[None, :], np.asarray(xf.full()).T))

        # jac:xf:p is d vec(xf) / d p, vec stacking xf's columns (one per
        # output time); at the start every sensitivity is zero.
        sens = np.asarray(jacobian.full()).reshape(grid.size, n, free.size)
        self._sensitivities = np.concatenate(
            (np.zeros((1, n, free.size)), sens))

        return np.concatenate(([start_time], grid)), states

    def _solve_dae(self, unit, states_init, start_time, grid, opts):
        """IDAS on the unit's differential and algebraic states."""

        ca = import_casadi()

        if self.locate_events and unit.compiled_events:
            raise NotImplementedError(
                "locate_events is not available with algebraic states.")

        algebraic = np.asarray(unit.solver_state_collection.algebraic_mask,
                               dtype=bool)
        differential = ~algebraic

        self._dae = _casadi_dae(self.model, algebraic)

        t = ca.MX.sym("t")
        x = ca.MX.sym("x", int(differential.sum()))
        z = ca.MX.sym("z", int(algebraic.sum()))
        ode, alg = self._dae(t, x, z)

        integrator = ca.integrator("pharmapy", "idas",
                                   {"t": t, "x": x, "z": z,
                                    "ode": ode, "alg": alg},
                                   start_time, list(grid), opts)

        result = integrator(x0=states_init[differential],
                            z0=states_init[algebraic])
        self._stats = integrator.stats()

        states = np.empty((len(grid) + 1, states_init.size))
        states[0] = states_init
        states[1:, differential] = np.asarray(result["xf"].full()).T
        states[1:, algebraic] = np.asarray(result["zf"].full()).T

        return np.concatenate(([start_time], grid)), states

    @property
    def statistics(self):
        return dict(self._stats)
