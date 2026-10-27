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

JAX and CasADi are optional: neither is imported until one of these backends
is constructed.
"""

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

    All methods take and return numpy arrays.
    """

    def __init__(self, unit, time, states):

        jax = import_jax()

        self.jax = jax
        self.n_states = int(np.size(states))
        self.start_time = float(time)
        self.traces = 0

        # How often a non-finite exact derivative was replaced by a
        # difference quotient (see jac and jvp).
        self.patched_columns = 0
        self.patched_products = 0

        # Calls made by the integrator, for comparing backends.
        self.counts = {"rhs": 0, "jac": 0, "jvp": 0}

        t0 = perf_counter()

        states = _state_argument(states)
        start = _time_argument(time)
        unit.unit_model(time, states)

        recorded = []
        discovery = unit.create_pseudo()

        with xp.tracing_at(self.start_time), xp.time_inputs(record=recorded):
            jax.eval_shape(
                lambda t, y: jax.numpy.asarray(discovery.unit_model(t, y)),
                start, states)

        self.input_shapes = [shape for shapes in recorded for shape in shapes]
        self.time_functions = []

        self.pseudo = unit.create_pseudo()
        pseudo = self.pseudo

        def rhs(time, states, inputs):
            # Runs only while JAX traces. A second trace would run the copy's
            # objects with values the first left behind, so it is refused.
            self.traces += 1

            if self.traces > 1:
                raise RuntimeError(
                    "The right-hand side was traced a second time. Call the "
                    "compiled function with float64 arguments of fixed "
                    "shape, as TracedModel's methods do."
                )

            with xp.time_inputs(supply=inputs) as supplied:
                rates = jax.numpy.asarray(pseudo.unit_model(time, states))

            if supplied.used != len(inputs):
                raise RuntimeError(
                    "The trace met different functions of time the second "
                    "time through, so they cannot be passed in as inputs.")

            self.time_functions = supplied.functions

            return rates

        self._rhs = jax.jit(rhs)
        self._jac = None
        self._jvp = None
        self._jac_time = None

        placeholder = [np.zeros(shape) for shape in self.input_shapes]

        with xp.tracing_at(self.start_time):
            self._rhs(start, states, placeholder).block_until_ready()

        self.trace_seconds = perf_counter() - t0
        self.jacobian_seconds = None

    # ------------------------------------------------------------------
    # Compiled functions
    # ------------------------------------------------------------------

    def inputs(self, time):
        """The functions of time the trace could not follow, at ``time``."""

        values = []

        for function in self.time_functions:
            leaves, _ = xp._flatten(function(float(time)))
            values += [np.asarray(leaf, dtype=np.float64) for leaf in leaves]

        return values

    def _arguments(self, time, states):
        return (_time_argument(time), _state_argument(states),
                self.inputs(time))

    def rhs(self, time, states):
        self.counts["rhs"] += 1

        return np.array(self._rhs(*self._arguments(time, states)),
                        dtype=float)

    def jac(self, time, states):
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
            self._jac(*self._arguments(time, states)).block_until_ready()

            self.jacobian_seconds = perf_counter() - t0

        self.counts["jac"] += 1

        jacobian = np.array(self._jac(*self._arguments(time, states)),
                            dtype=float)

        bad = ~np.isfinite(jacobian).all(axis=0)

        if bad.any():
            states = _state_argument(states)
            base = self.rhs(time, states)

            for column in np.flatnonzero(bad):
                step = self._difference_step(states[column])
                shifted = states.copy()
                shifted[column] += step
                jacobian[:, column] = (self.rhs(time, shifted) - base) / step

            self.patched_columns += int(bad.sum())

        return jacobian

    def jvp(self, time, states, vector):
        """Exact d(rhs)/d(states) times a vector, without forming the matrix.

        Falls back to a directional difference when the exact product is not
        finite, for the reason given in jac.
        """

        if self._jvp is None:
            jax = self.jax
            rhs = self._rhs

            def jvp(time, states, inputs, vector):
                return jax.jvp(lambda y: rhs(time, y, inputs), (states,),
                               (vector,))[1]

            self._jvp = jax.jit(jvp)

        states = _state_argument(states)
        vector = _state_argument(vector)

        self.counts["jvp"] += 1

        product = np.array(
            self._jvp(*self._arguments(time, states), vector), dtype=float)

        if not np.isfinite(product).all():
            norm = np.linalg.norm(vector)

            if norm > 0:
                step = self._difference_step(np.linalg.norm(states)) / norm
                product = (self.rhs(time, states + step * vector)
                           - self.rhs(time, states)) / step

            self.patched_products += 1

        return product

    @staticmethod
    def _difference_step(value):
        return np.sqrt(np.finfo(float).eps) * max(1.0, abs(float(value)))

    def jac_time(self, time, states):
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
                    compiled(*self._arguments(time, states)
                             ).block_until_ready()
                    self._jac_time = compiled
                except Exception:
                    pass

        if self._jac_time:
            return np.array(self._jac_time(*self._arguments(time, states)),
                            dtype=float).reshape(-1, 1)

        step = np.sqrt(np.finfo(float).eps) * max(1.0, abs(float(time)))

        return ((self.rhs(time + step, states) - self.rhs(time, states))
                / step).reshape(-1, 1)
    # ------------------------------------------------------------------
    # CasADi
    # ------------------------------------------------------------------

    def casadi_function(self, name="pharmapy_rhs"):
        """
        The compiled right-hand side as a casadi.Function f(t, x) -> dx/dt.

        Its derivatives are the exact ones above, so an integrator, a
        collocation transcription or an NLP built on it gets exact Jacobians
        from JAX. The returned function keeps the objects behind it alive.
        """

        return _casadi_rhs(self, name)


def _casadi_rhs(model, name):
    """casadi.Callback pair: f(t, x) and its Jacobian, evaluated by JAX."""

    ca = import_casadi()
    n = model.n_states

    class Jacobian(ca.Callback):

        def __init__(self, name, opts):
            ca.Callback.__init__(self)
            self.construct(name, opts)

        # Inputs: t, x, and the nominal output CasADi passes along.
        def get_n_in(self):
            return 3

        def get_n_out(self):
            return 2

        def get_sparsity_in(self, index):
            return ca.Sparsity.dense((1, n, n)[index], 1)

        def get_sparsity_out(self, index):
            return ca.Sparsity.dense(n, (1, n)[index])

        def eval(self, arg):
            time = float(arg[0])
            states = np.ravel(arg[1].full())

            return [model.jac_time(time, states), model.jac(time, states)]

    class Forward(ca.Callback):
        """Directional derivatives.

        A few directions -- one per Krylov iteration, which is what CasADi's
        GMRES Newton scheme asks for -- are one JAX jvp each, far cheaper
        than forming the Jacobian. CasADi also builds the Jacobian for its
        direct solve out of forward directions, all n of them at once; that
        request is the exact Jacobian times the seeds.
        """

        def __init__(self, n_directions, name, opts):
            ca.Callback.__init__(self)
            self.n_directions = n_directions
            self.construct(name, opts)

        # Inputs: t, x, the nominal output, then the seeds for t and x.
        def get_n_in(self):
            return 5

        def get_n_out(self):
            return 1

        def get_sparsity_in(self, index):
            k = self.n_directions
            return ca.Sparsity.dense(*((1, 1), (n, 1), (n, 1), (1, k),
                                       (n, k))[index])

        def get_sparsity_out(self, index):
            return ca.Sparsity.dense(n, self.n_directions)

        def eval(self, arg):
            time = float(arg[0])
            states = np.ravel(arg[1].full())
            seed_time = np.ravel(arg[3].full())
            seed_states = np.asarray(arg[4].full()).reshape(
                n, self.n_directions)

            by_time = (model.jac_time(time, states).ravel()
                       if np.any(seed_time) else None)

            if self.n_directions > max(1, n // 16):
                out = model.jac(time, states) @ seed_states
            else:
                out = np.zeros((n, self.n_directions))

                for k in range(self.n_directions):

                    if np.any(seed_states[:, k]):
                        out[:, k] = model.jvp(time, states,
                                              seed_states[:, k])

            if by_time is not None:
                out = out + np.outer(by_time, seed_time)

            return [out]

    class RHS(ca.Callback):

        def __init__(self, name, opts):
            ca.Callback.__init__(self)
            self.construct(name, opts)

        def get_n_in(self):
            return 2

        def get_n_out(self):
            return 1

        def get_name_in(self, index):
            return ("t", "x")[index]

        def get_name_out(self, index):
            return "ode"

        def get_sparsity_in(self, index):
            return ca.Sparsity.dense((1, n)[index], 1)

        def get_sparsity_out(self, index):
            return ca.Sparsity.dense(n, 1)

        def eval(self, arg):
            return [model.rhs(float(arg[0]), np.ravel(arg[1].full()))]

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

class JaxBackend(ScipyBackend):
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

    def compile_integrator(self, unit, eval_sens=False, jac_v_prod=False,
                           options=None, verbose=True, any_event=True):

        states_init = super().compile_integrator(
            unit, eval_sens, jac_v_prod, options, verbose, any_event)

        self.model = TracedModel(unit, unit.elapsed_time, states_init)

        return states_init

    def evaluate_rhs(self, unit, time, states):
        return self.model.rhs(time, states)

    def translate_options(self):

        translated = super().translate_options()

        if self.exact_jacobian and self.method in self.IMPLICIT_METHODS:
            translated.setdefault("jac", self.model.jac)

        return translated


class JaxAssimuloBackend(AssimuloBackend):
    """
    AssimuloBackend (CVode) on the JAX-compiled right-hand side.

    CVode gets the compiled right-hand side and exact derivatives of it, in
    the form its linear solver uses: the dense Jacobian under DENSE, and
    Jacobian-vector products under SPGMR. The difference quotients CVode
    would otherwise form -- n evaluations per Jacobian, or one per Krylov
    iteration -- are not needed. A crystallizer asks for SPGMR
    (linear_solver_hint = "krylov"); that is followed from KRYLOV_MIN_STATES
    states up, and below that the exact dense Jacobian is the faster choice.

    Events work exactly as under AssimuloBackend: they are evaluated on the
    unit itself.

    Parameters
    ----------
    exact_jacobian : bool, optional
        Hand CVode the exact derivatives. Default True. False keeps only the
        compiled right-hand side.
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

    def compile_integrator(self, unit, eval_sens=False, jac_v_prod=False,
                           options=None, verbose=True, any_event=True):

        if eval_sens:
            raise NotImplementedError(
                "JaxAssimuloBackend does not compute parameter "
                "sensitivities yet. Use AssimuloBackend for them."
            )

        states_init = super().compile_integrator(
            unit, eval_sens, jac_v_prod, options, verbose, any_event)

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

        self.model = TracedModel(unit, unit.elapsed_time, states_init)
        model = self.model

        def rhs(time, states, sw=None):
            return model.rhs(time, states)

        events = unit.compiled_events

        if events:
            problem = Explicit_Problem(rhs, states_init,
                                       t0=unit.elapsed_time,
                                       sw0=[True] * len(events))
            attach_assimulo_events(problem, unit)
        else:
            problem = Explicit_Problem(rhs, states_init,
                                       t0=unit.elapsed_time)

        if self.exact_jacobian:

            def jac(time, states, sw=None):
                return model.jac(time, states)

            def jacv(time, states, rhs_value, vector, sw=None):
                return model.jvp(time, states, vector)

            problem.jac = jac
            problem.jacv = jacv

        self._problem = problem


class CasadiBackend(IntegratorBackend):
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

    What it does not do, and refuses rather than fakes:

    - **Terminal events.** CasADi restarts at a crossing but this backend has
      no way yet to end the run there, so a unit with an event that ends the
      run is refused.
    - **Sensitivities**, for now.

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
                           options=None, verbose=True, any_event=True):

        if eval_sens:
            raise NotImplementedError(
                "CasadiBackend does not compute sensitivities yet."
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

        self.model = TracedModel(unit, unit.elapsed_time, states_init)
        self.function = self.model.casadi_function()

        unit.configure_solver()

        self._compiled = True

        return states_init

    def solve(self, unit, runtime=None, time_grid=None, eval_sens=False,
              jac_v_prod=False, verbose=True, options=None, any_event=True):

        if not self._compiled:
            self.compile_integrator(unit, eval_sens, jac_v_prod, options,
                                    verbose, any_event)

        unit.derivatives = np.array(
            unit.unit_model(time=unit.elapsed_time,
                            states=unit.create_solver_init_states()),
            dtype=float)

        time, states = self.fast_solve(unit, runtime=runtime,
                                       time_grid=time_grid, verbose=verbose)

        unit.retrieve_results(time, states)

        return time, states

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
