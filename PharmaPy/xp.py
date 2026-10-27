"""
Array namespace for the right-hand-side path.

The modules a vessel's right-hand side runs through import this module as
``np``. Outside a trace it *is* numpy: ``np.where`` here is numpy's own
function, found in this module's namespace, so numeric callers get the same
results at the same cost as before. While a trace is being built
(``with tracing_at(time):``, which TracedModel does), each name is swapped
for a dispatcher that sends a call with a JAX tracer among its arguments to
jax.numpy, so the right-hand side is recorded rather than evaluated.

Two rules keep code on that path traceable:

- write into an array with ``set_at`` / ``add_at``, which return the result
  (numpy writes in place, as before; a traced array is immutable);
- branch on a computed value only when ``not is_traced(value)``. A trace
  needs both outcomes as array expressions, usually through ``where``.

``is_traced`` checks the type, inside a trace or not, so a branch that must
not see a tracer never does. Tracing outside ``tracing_at`` fails at the first
numpy call that meets a tracer.

JAX is never imported here. A tracer can only exist once something else has
imported JAX.
"""

import sys

import numpy as _np

_tracer_type = None


def _tracer():
    global _tracer_type

    if _tracer_type is None:
        jax = sys.modules.get("jax")

        if jax is None:
            return None

        try:
            _tracer_type = jax.core.Tracer
        except AttributeError:  # should a JAX release stop exporting it
            from jax._src.core import Tracer as _tracer_type

    return _tracer_type


def _contains(obj, tracer):
    if isinstance(obj, tracer):
        return True

    if isinstance(obj, (list, tuple)):
        for item in obj:
            if _contains(item, tracer):
                return True

    elif isinstance(obj, dict):
        for item in obj.values():
            if _contains(item, tracer):
                return True

    return False


def is_traced(obj):
    """True if obj, or anything in a list, tuple or dict of it, is a tracer."""
    tracer = _tracer_type

    if tracer is None:
        # Called on every branch the right-hand side takes, so the common
        # case -- no JAX anywhere in the process -- is answered inline.
        if "jax" not in sys.modules:
            return False

        tracer = _tracer()

    return _contains(obj, tracer)


def _jnp():
    import jax.numpy

    return jax.numpy


def _dispatcher(name, numpy_attr):

    def dispatch(*args, **kwargs):
        if is_traced(args) or (kwargs and is_traced(kwargs)):
            return getattr(_jnp(), name)(*args, **kwargs)

        return numpy_attr(*args, **kwargs)

    dispatch.__name__ = name
    dispatch.__doc__ = numpy_attr.__doc__

    return dispatch


# name -> (numpy's function, its dispatcher), for every function looked up.
_functions = {}

# How many tracing_at contexts are open.
_tracing = 0


def __getattr__(name):
    attr = getattr(_np, name)

    # Constants, dtypes and submodules are the numpy objects themselves.
    if not callable(attr) or isinstance(attr, type):
        globals()[name] = attr
        return attr

    _functions[name] = (attr, _dispatcher(name, attr))

    # Placed in the module namespace, so later lookups never come back
    # through here: numpy's function outside a trace, the dispatcher inside.
    current = _functions[name][_tracing > 0]
    globals()[name] = current

    return current


def _install(dispatching):
    namespace = globals()

    for name, pair in _functions.items():
        namespace[name] = pair[dispatching]


def as_float(value):
    """float(value) for a number; a traced value passes through unchanged."""
    if is_traced(value):
        return value

    return float(value)


def set_at(array, index, value):
    """array[index] = value, returning the result.

    numpy: writes into ``array`` in place and returns it, exactly as before.
    Traced: returns a new array, promoting a numpy buffer that is about to
    receive a traced value.
    """
    if is_traced((array, value)):
        return _jnp().asarray(array).at[index].set(value)

    array[index] = value

    return array


def add_at(array, index, value):
    """array[index] += value, returning the result (see set_at)."""
    if is_traced((array, value)):
        return _jnp().asarray(array).at[index].add(value)

    array[index] += value

    return array


# A time at which functions of time can be evaluated while a trace is being
# built, to learn the shape of what they return. Set by the backend that
# traces (see tracing_at); the start of the run is always a valid choice.
_probe_time = 0.0


class tracing_at:
    """Context for building a trace that starts at ``time``.

    Inside it, the numpy names in this module dispatch tracers to jax.numpy;
    on leaving it they are numpy's own functions again. ``of_time`` evaluates
    the functions of time it meets once, at ``time``, to learn the shapes of
    their results.

    Only building the trace needs this. Running the compiled function does
    not go back through the Python objects.
    """

    def __init__(self, time):
        self.time = float(time)

    def __enter__(self):
        global _probe_time, _tracing
        self._previous = _probe_time
        _probe_time = self.time

        _tracing += 1

        if _tracing == 1:
            _install(dispatching=True)

        return self

    def __exit__(self, *exc):
        global _probe_time, _tracing
        _probe_time = self._previous

        _tracing -= 1

        if _tracing == 0:
            _install(dispatching=False)

        return False


def _flatten(tree):
    """Leaves of nested dicts/lists/tuples in order, and how to rebuild them.

    Written out rather than borrowed from jax.tree_util, which sorts dict
    keys: the inlet trajectories are keyed by tuples holding PhaseRefs, and
    those do not order.
    """
    if isinstance(tree, dict):
        keys = list(tree)
        parts = [_flatten(tree[key]) for key in keys]
        leaves = [leaf for part in parts for leaf in part[0]]
        sizes = [len(part[0]) for part in parts]
        return leaves, ("dict", keys, [part[1] for part in parts], sizes)

    if isinstance(tree, (list, tuple)):
        parts = [_flatten(item) for item in tree]
        leaves = [leaf for part in parts for leaf in part[0]]
        sizes = [len(part[0]) for part in parts]
        return leaves, (type(tree), None, [part[1] for part in parts], sizes)

    if tree is None:
        return [], None

    return [tree], "leaf"


def _unflatten(leaves, spec):
    if spec is None:
        return None

    if spec == "leaf":
        return leaves[0]

    kind, keys, specs, sizes = spec
    items = []
    start = 0

    for sub_spec, size in zip(specs, sizes):
        items.append(_unflatten(leaves[start:start + size], sub_spec))
        start += size

    if kind == "dict":
        return dict(zip(keys, items))

    return kind(items)


def of_time(function, time):
    """``function(time)`` for a function of time alone.

    A function a trace can follow -- a constant, arithmetic on ``t`` -- is
    traced like the rest of the model, which costs nothing when the compiled
    function runs.

    Temperature profiles and recorded upstream trajectories are usually
    ordinary Python (interpolants, ``if t < t_switch``), which a trace cannot
    look inside, and they raise on the tracer. They do not depend on the
    state, so nothing is lost by running them on the number. Inside a
    ``time_inputs`` context (TracedModel) their values become inputs of the
    compiled function, evaluated before each call. Otherwise the call goes
    through ``jax.pure_callback``, which runs ``function`` on the host each
    time the compiled function runs, at about 0.2 ms a call. The result may
    be a number, an array or nested dicts and lists of them.

    The derivative with respect to the state is unaffected either way. One
    with respect to time is not available through an input or a callback;
    TracedModel.jac_time falls back to a difference quotient for it.
    """
    if not is_traced(time):
        return function(time)

    import jax

    try:
        return function(time)
    except Exception:
        # Whatever stopped the trace, the function is run on the number
        # below, where a genuine error raises again.
        pass

    template = function(_probe_time)
    leaves, spec = _flatten(template)

    if not leaves:
        return template

    if _inputs is not None:
        return _inputs.stand_in(function, template, leaves, spec)

    shapes = [jax.ShapeDtypeStruct(_np.shape(leaf), _np.float64)
              for leaf in leaves]

    def host(value):
        result_leaves, result_spec = _flatten(function(float(value)))

        if result_spec != spec:
            raise RuntimeError(
                f"{getattr(function, '__name__', function)!r} returned a "
                f"different structure at t={float(value):g} than at "
                f"t={_probe_time:g}, so it cannot be compiled into the "
                "right-hand side.")

        return [_np.asarray(leaf, dtype=_np.float64) for leaf in result_leaves]

    results = jax.pure_callback(host, shapes, time)

    return _unflatten(list(results), spec)


# The time_inputs context in force, if any; see of_time.
_inputs = None


class time_inputs:
    """The functions of time a trace cannot follow, as inputs to it.

    A host callback costs about 0.2 ms every time the compiled function runs,
    more than many right-hand sides cost in all. TracedModel avoids it by
    tracing twice:

    ``record=[]``
        Each such function the trace meets is noted, in order, with the
        shapes of its result, and its value at the probe time stands in for
        it. Only the shapes matter in this pass.
    ``supply=[...]``
        The same calls, in the same order, take their values from this flat
        list of traced arguments of the compiled function. ``functions``
        collects the functions met, to be evaluated on the number before each
        call; ``used`` counts the arguments consumed.
    """

    def __init__(self, record=None, supply=None):
        self.record = record
        self.supply = supply
        self.functions = []
        self.used = 0

    def stand_in(self, function, template, leaves, spec):

        if self.record is not None:
            self.record.append([_np.shape(leaf) for leaf in leaves])
            return template

        start = self.used
        self.used += len(leaves)

        if self.used > len(self.supply):
            raise RuntimeError(
                "The trace met more functions of time than when they were "
                "recorded, so they cannot be passed in as inputs.")

        self.functions.append(function)

        return _unflatten(list(self.supply[start:self.used]), spec)

    def __enter__(self):
        global _inputs
        self._previous = _inputs
        _inputs = self
        return self

    def __exit__(self, *exc):
        global _inputs
        _inputs = self._previous
        return False
