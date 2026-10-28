"""
Numbers of a unit declared as parameters.

A declaration names numbers that live on objects inside a unit -- the
kinetic constants of its RxnKinetics or CrystKinetics, a controller's set
point -- and says how to read and write them. Two things use it:

- The traced backends (PharmaPy.TracedBackends) make declared numbers inputs
  of the compiled model rather than constants frozen into it, so new values
  need no new trace.
- Sensitivities (solve_unit(eval_sens=True)) are taken with respect to the
  declared numbers marked free.

Values are always read from, and written to, the objects themselves, so
``kinetics.set_params(...)`` and ``parameters.assign(...)`` do the same
thing.

    parameters = Parameters.kinetics(reactor)    # every kinetics object
    parameters.free = [True] * 4 + [False] * 4   # what sensitivities cover
    parameters.assign(theta)
    reactor.reset()
    reactor.solve_unit(time_grid=t, parameters=parameters, eval_sens=True)
    reactor.result.sensitivities['mole_conc_liquid0']   # (time, species, free)
"""

import copy

import numpy as np

from PharmaPy import xp


class ParameterBlock:
    """The parameters held by one object, and how to read and write them.

    ``read(owner)`` returns the values as a flat array; ``write(owner,
    values)`` stores a flat array of the same size. Under a trace ``values``
    is a traced array, so ``write`` must not convert it to numbers.
    """

    def __init__(self, name, owner, read, write, names=None, size=None):

        self.name = name
        self.owner = owner
        self.read = read
        self.write = write

        if size is None:
            size = self.values().size

        self.size = int(size)

        if names is None or len(names) != self.size:
            names = [f"{name}[{index}]" for index in range(self.size)]

        self.names = list(names)

    def values(self):
        return np.asarray(self.read(self.owner), dtype=float).reshape(-1)

    def assign(self, values):
        self.write(self.owner, values)

    def on(self, owner):
        """The same block, reading and writing another object."""

        return ParameterBlock(self.name, owner, self.read, self.write,
                              self.names, self.size)


def _read_kinetics(kinetics):
    return kinetics.concat_params()


def _write_kinetics(kinetics, values):
    kinetics.set_params(values)


def _attribute_access(attribute, shape):

    def read(owner):
        return getattr(owner, attribute)

    def write(owner, values):
        value = values.reshape(shape)

        # A number stays a number for numeric callers; under a trace the
        # value is a traced array whatever its shape.
        if shape == () and not xp.is_traced(value):
            value = float(value)

        setattr(owner, attribute, value)

    return read, write


def find_kinetics(unit):
    """Every kinetics object a unit's mechanisms use, each once, in order.

    A kinetics object is what a mechanism keeps as ``kinetics``
    (ReactionMechanism) or ``mechanism_kinetics`` (the population balances).
    """

    found = []
    seen = set()

    for mechanism in unit.iter_mechanisms():

        for attribute in ("kinetics", "mechanism_kinetics"):
            kinetics = getattr(mechanism, attribute, None)

            if (kinetics is None or id(kinetics) in seen
                    or not hasattr(kinetics, "concat_params")):
                continue

            seen.add(id(kinetics))
            found.append(kinetics)

    return found


class Parameters:
    """
    Declared parameters of a unit, as one flat vector.

    The vector is the blocks' values concatenated in the order the blocks
    were added. ``free`` marks the entries sensitivities are taken with
    respect to (all of them by default); a fixed entry is still an input of
    the compiled model.
    """

    def __init__(self, blocks=(), free=None):

        self.blocks = list(blocks)
        self._free = None

        if free is not None:
            self.free = free

    # ------------------------------------------------------------------
    # Declaring
    # ------------------------------------------------------------------

    @classmethod
    def kinetics(cls, unit, free=None):
        """Every kinetics object in ``unit``, in the order its mechanisms
        hold them.

        For a ConcurrentVessel, every member's, each block named after its
        member. The values are what the kinetics' concat_params returns
        (the reformulated phi_1, phi_2 when reformulate_kin is set), as for
        legacy parameter estimation.
        """

        parameters = cls()

        members = getattr(unit, "members", None)
        sources = (list(members.items()) if isinstance(members, dict)
                   else [(None, unit)])

        for member, source in sources:

            for kinetics in find_kinetics(source):
                name = type(kinetics).__name__

                if member is not None:
                    name = f"{member}.{name}"

                parameters.add_kinetics(kinetics, name=name)

        if not parameters.blocks:
            raise ValueError(
                f"{type(unit).__name__} has no kinetics objects to declare "
                "as parameters.")

        if free is not None:
            parameters.free = free

        return parameters

    def add_kinetics(self, kinetics, name=None):
        """Declare a kinetics object's parameters (concat_params order)."""

        name = self._unique(name or type(kinetics).__name__)
        values = np.asarray(kinetics.concat_params(), dtype=float)

        # set_params has to be the inverse of concat_params, or assigning a
        # vector would put values in the wrong places.
        probe = copy.deepcopy(kinetics)
        probe.set_params(values.copy())
        round_trip = np.asarray(probe.concat_params(), dtype=float)

        if round_trip.shape != values.shape or not np.allclose(
                round_trip, values, rtol=1e-12, atol=0, equal_nan=True):
            raise ValueError(
                f"{type(kinetics).__name__}.set_params does not restore what "
                "concat_params returns, so its parameters cannot be "
                "declared.")

        self.blocks.append(ParameterBlock(
            name, kinetics, _read_kinetics, _write_kinetics,
            names=getattr(kinetics, "name_params", None), size=values.size))
        self._free = None

        return self

    def add(self, name, owner, attribute=None, read=None, write=None,
            names=None):
        """
        Declare numbers held by ``owner``.

        Either an ``attribute`` of it (a number or an array, written back in
        the same shape), or a ``read(owner)`` / ``write(owner, values)``
        pair working with flat arrays. ``owner`` must be an object inside
        the unit -- its controller, a mechanism, a kinetics object -- so the
        compiled model can find its copy. Returns self, so declarations
        chain.
        """

        if attribute is not None:
            shape = np.shape(getattr(owner, attribute))
            read, write = _attribute_access(attribute, shape)

            if names is None and shape == ():
                names = [name]

        elif read is None or write is None:
            raise ValueError("Give an attribute, or both read and write.")

        self.blocks.append(ParameterBlock(self._unique(name), owner, read,
                                          write, names=names))
        self._free = None

        return self

    def _unique(self, name):
        taken = {block.name for block in self.blocks}

        if name not in taken:
            return name

        index = 2
        while f"{name}_{index}" in taken:
            index += 1

        return f"{name}_{index}"

    # ------------------------------------------------------------------
    # The vector
    # ------------------------------------------------------------------

    @property
    def size(self):
        return sum(block.size for block in self.blocks)

    @property
    def names(self):
        if len(self.blocks) == 1:
            return list(self.blocks[0].names)

        return [f"{block.name}:{name}" for block in self.blocks
                for name in block.names]

    @property
    def free(self):
        """Boolean mask: the entries sensitivities are taken for."""

        if self._free is None:
            return np.ones(self.size, dtype=bool)

        return self._free

    @free.setter
    def free(self, mask):
        mask = np.asarray(mask, dtype=bool).reshape(-1)

        if mask.size != self.size:
            raise ValueError(
                f"free has {mask.size} entries for {self.size} parameters.")

        self._free = mask

    @property
    def free_index(self):
        return np.flatnonzero(self.free)

    @property
    def free_names(self):
        names = self.names
        return [names[index] for index in self.free_index]

    def values(self):
        """The current values, read from the owners."""

        if not self.blocks:
            return np.zeros(0)

        return np.concatenate([block.values() for block in self.blocks])

    def free_values(self):
        return self.values()[self.free]

    def assign(self, values):
        """Write a full vector to the owners.

        Under a trace ``values`` is the compiled function's traced input,
        and the owners are the copies the trace runs through (see bind).
        """

        if not xp.is_traced(values):
            values = np.asarray(values, dtype=float).reshape(-1)

            if values.size != self.size:
                raise ValueError(
                    f"{values.size} values for {self.size} parameters.")

        start = 0

        for block in self.blocks:
            part = values[start:start + block.size]

            # Owners keep what they are given (set_params stores slices of
            # it), so a numeric vector is copied rather than shared.
            if not xp.is_traced(part):
                part = part.copy()

            block.assign(part)
            start += block.size

    def assign_free(self, values):
        """Write the free entries, leaving the fixed ones as they are."""

        full = self.values()
        full[self.free] = np.asarray(values, dtype=float).reshape(-1)
        self.assign(full)

    def full(self, free_values, base=None):
        """A full vector: ``base`` (default: current values) with the free
        entries replaced."""

        full = self.values() if base is None else np.array(base, dtype=float)
        full[self.free] = free_values

        return full

    # ------------------------------------------------------------------
    # Copies of the unit
    # ------------------------------------------------------------------

    def bind(self, memo):
        """
        The same declaration, on the copies a deepcopy ``memo`` made.

        Tracing runs a copy of the unit (create_pseudo), so the compiled
        model has to write its parameter inputs to the copies of the owners,
        not to the owners themselves.
        """

        blocks = []

        for block in self.blocks:
            owner = memo.get(id(block.owner))

            if owner is None:
                raise ValueError(
                    f"Parameter block {block.name!r}: its owner, a "
                    f"{type(block.owner).__name__}, is not part of the unit "
                    "being compiled, so the compiled model cannot set it.")

            blocks.append(block.on(owner))

        return Parameters(blocks, free=self._free)

    def signature(self):
        """What a compiled model depends on: blocks, owners, sizes, free."""

        return (tuple((block.name, id(block.owner), block.size)
                      for block in self.blocks),
                tuple(self.free))

    def __repr__(self):
        return (f"Parameters({self.size} values, {int(self.free.sum())} "
                f"free: {', '.join(self.free_names)})")
