"""
Several MultiPhaseVessels integrated as one system.

SimulationExec solves unit operations one after another, each over its whole
time span, and hands the next one the trajectory the last one produced. That
cannot represent a recycle -- the upstream unit would need a trajectory that
does not exist yet -- and a controller on one unit cannot act on a measurement
taken in another. A ConcurrentVessel instead hands all of its members to ONE
integrator as one state vector, so every member advances through each time
step together and each reads the live outlet of the members feeding it.

To an IntegratorBackend it is simply another unit. It offers the methods a
MultiPhaseVessel offers a backend (unit_model, create_solver_init_states,
retrieve_results, the event methods, ...) and follows the same logic behind
them, so every backend integrates it unchanged.

Recycle
-------
A link that feeds a member evaluated earlier closes a loop; it is a tear. The
loop is algebraic only through flow: an outlet's composition and temperature
are its vessel's own state, but a level controller that sets
outlet = inlet + K * (V - V_target) makes each outlet flow depend on the
inlet flow. There are two ways to close it:

``tear_mode='iterate'`` (default)
    The tear is converged inside every right-hand side evaluation, by
    Wegstein-accelerated substitution. Works on every backend, scipy
    included. With no recycle there is nothing to iterate, and a single pass
    in evaluation order is exact.
``tear_mode='algebraic'``
    The tear values become algebraic solver states, and the integrator solves
    for them along with everything else. Needs a backend that handles
    differential-algebraic systems (AssimuloDAEBackend, DiffeqpyBackend).

If iteration fails to converge, the solve restarts with algebraic tears on a
DAE backend -- the configured integrator if it is one, otherwise
``dae_integrator`` -- and ``requires_dae`` is set. With neither available it
stops with a TearConvergenceError that says so.

Control across units
--------------------
Before any balance is evaluated, every member is brought to the state being
evaluated. A member's controller may therefore hold a reference to another
member of the same ConcurrentVessel and read its state -- for instance,
manipulate a feed from a concentration measured downstream. The member still
has exactly one controller; that controller just observes a neighbour. The
results replay runs every member together, so what such a controller saw
during the run is what it sees during the replay.

Flowsheets
----------
A ConcurrentVessel can be a single node of a SimulationExec flowsheet. It is
fed through one inlet of its ``entry`` member, and it discharges what its
``exit`` member's outlet does not send round a link.

Example
-------
A reactor pair with half of the second reactor's outlet recycled::

    cv = ConcurrentVessel(
        {'R01': reactor_1, 'R02': reactor_2},
        links=[StreamLink('R01', 'R02'),
               StreamLink('R02', 'R01', fraction=0.5)],
        integrator=AssimuloBackend())

    cv.solve_unit(runtime=3600)
    cv.R02.result.mole_conc_liquid0
"""

import copy
import warnings
from contextlib import contextmanager

from PharmaPy import xp as np
from PharmaPy.DataClasses import StateCollection, StateVariable
from PharmaPy.IntegratorBackends import ScipyBackend
from PharmaPy.MultiPhaseVessel import MultiPhaseVessel


class TearConvergenceError(RuntimeError):
    """A recycle could not be converged by iterating inside the RHS."""


class StreamLink:
    """
    One stream from a member's outlet into another member's inlet.

    Parameters
    ----------
    source, sink : str
        Member names. The source must be a continuous unit, since only a
        continuous unit discharges a stream while it runs.
    fraction : float, optional
        Share of the source's outlet flow sent along this link, in (0, 1].
        Whatever share no link takes leaves the ConcurrentVessel: it is the
        product if the source is the exit member, and a purge otherwise.
    inlet : int, optional
        The sink's inlet connection this link feeds. The default, None, adds
        a new connection built from the source's outlet stream and leaves
        every configured feed in place. An index replaces that connection --
        a placeholder feed, for instance.
    name : str, optional
        Label for the link. Defaults to 'source->sink'.
    """

    def __init__(self, source, sink, fraction=1.0, inlet=None, name=None):

        self.source = source
        self.sink = sink
        self.fraction = float(fraction)
        self.inlet = inlet
        self.name = name

        # Filled in by ConcurrentVessel.
        self.coupling = None
        self.inlet_index = None
        self.is_tear = False

    @property
    def label(self):
        if self.name is not None:
            return self.name

        return f"{self.source}->{self.sink}"

    def __repr__(self):
        share = "" if self.fraction == 1.0 else f", fraction={self.fraction:g}"
        tear = ", tear" if self.is_tear else ""

        return f"StreamLink({self.source!r} -> {self.sink!r}{share}{tear})"


class CoupledInlet:
    """
    A live inlet: another member's outlet, plugged in as a DynamicInlet.

    MultiPhaseVessel._inlet_values_at already asks a stream's DynamicInlet
    for the inlet values at each evaluation, so a coupling needs no change to
    how a vessel resolves its inlets. It serves, in order of precedence:

    frozen
        Pinned while the coordinator evaluates outputs, events or final
        conditions, so every member sees exactly the values the right-hand
        side used, whatever else is re-resolved meanwhile.
    override
        The tear value of a link that closes a recycle: the guess being
        iterated, or the algebraic state the integrator carries.
    live
        The source's outlet_values(), valid once the source has been
        evaluated at the current point.

    Whatever the source, every value is served clipped at zero: a stream
    carries no negative amounts. That matters more here than between
    sequential units, because the integrator's finite-difference Jacobian
    perturbs one member's state and the perturbation now reaches the next
    member within the same evaluation. A trace species perturbed slightly
    below zero in the source would otherwise arrive as a negative inflow,
    and the receiving vessel's positivity limiter -- one factor per phase,
    set by its scarcest species -- would shut that whole phase's outlet over
    1e-22 kg of a species it barely holds, making the right-hand side jump.
    The price is round-off-sized: while a source inventory is negative, the
    source discharges that negative amount and the sink receives none.
    """

    # What this inlet serves depends on the source vessel's state, not on
    # time alone, so a traced right-hand side evaluates it in the trace
    # (MultiPhaseVessel._inlet_trajectories_traced).
    follows_state = True

    def __init__(self, source, fraction=1.0):

        self.source = source
        self.fraction = float(fraction)

        self.override = None
        self.frozen = None
        self.last_served = None

        # BaseStream's DynamicInlet setter writes these.
        self.controllable = ()
        self.parent_instance = None

    def live_values(self):
        if self.source is None:
            raise RuntimeError(
                "This coupled inlet was copied without the vessel that feeds "
                "it, so it has no live outlet to read. Solve the vessel "
                "through its ConcurrentVessel."
            )

        values = self.source.outlet_values()

        if 'vol_flow' in values:
            values['vol_flow'] = values['vol_flow'] * self.fraction

        return self.physical(values)

    @staticmethod
    def physical(values):
        """The values with anything negative raised to zero."""

        return {
            name: (np.maximum(value, 0.0)
                   if isinstance(value, np.ndarray) or np.is_traced(value)
                   else max(float(value), 0.0))
            for name, value in values.items()
        }

    def evaluate_inputs(self, time):

        if self.frozen is not None:
            values = self.frozen
        elif self.override is not None:
            # An algebraic tear state may sit below zero between Newton
            # iterations; what arrives is still a physical stream.
            values = self.physical(self.override)
        else:
            values = self.live_values()

        self.last_served = values

        # Copies, because a mechanism may keep the array it is handed.
        return {
            name: value.copy() if isinstance(value, np.ndarray) else value
            for name, value in values.items()
        }

    def __deepcopy__(self, memo):
        copied = CoupledInlet.__new__(CoupledInlet)
        memo[id(self)] = copied

        for name, value in self.__dict__.items():

            if name == 'source':
                # Follow the source only into a joint copy that already holds
                # it (a ConcurrentVessel replay). Copying it alone would drag
                # a whole second vessel along, integrator included, for an
                # inlet that could not serve live values from it anyway.
                copied.source = memo.get(id(value))
            else:
                copied.__dict__[name] = copy.deepcopy(value, memo)

        return copied


class ConcurrentResult:
    """
    The results of a ConcurrentVessel solve.

    Each member's DynamicResult is reachable by name (``result.R01``, or
    ``result.members['R01']``) and is also left on the member itself, just
    as a standalone solve would leave it. ``couplings`` records what every
    linked inlet received, keyed by link label, on the shared time vector.
    """

    def __init__(self, time, members, couplings):

        self.time = np.asarray(time)
        self.members = dict(members)
        self.couplings = couplings

        for name, result in self.members.items():
            if name not in ('time', 'members', 'couplings'):
                setattr(self, name, result)

    def __repr__(self):
        lines = ['PharmaPy concurrent result', '-' * 26, '']

        for name, result in self.members.items():
            fields = sorted(
                key for key in vars(result)
                if key not in ('di_states', 'di_fstates', 'time')
            )
            lines.append(f"result.{name}: {', '.join(fields)}")

        if self.couplings:
            lines.append('')
            lines.append('result.couplings: ' + ', '.join(self.couplings))

        lines.append('')
        lines.append('Time vector can be accessed as result.time')

        return '\n'.join(lines)


class ConcurrentVessel:
    """
    Several MultiPhaseVessels solved as one system. See the module docstring.

    Parameters
    ----------
    members : dict
        {name: vessel}. Names must be valid identifiers; each member is also
        reachable as an attribute (``cv.R01``). Configure every member's own
        feeds before building the ConcurrentVessel: replacing a member's
        inlets afterwards disconnects its links.
    links : sequence of StreamLink, optional
        The streams between members. A tuple (source, sink) or
        (source, sink, fraction) is accepted in place of a StreamLink.
    integrator : IntegratorBackend, optional
        Integrates the whole system. Defaults to a ScipyBackend.
    tear_mode : {'iterate', 'algebraic'}, optional
        How recycles are closed. Default 'iterate'.
    order : sequence of str, optional
        Evaluation order. By default the members are visited depth-first in
        the order given, which evaluates an acyclic system upstream first and
        tears each loop on the link that returns to an earlier member.
    entry, entry_inlet : optional
        The member, and its inlet connection, that a flowsheet feed arrives
        through. Default: the first member, at its first inlet that no link
        feeds (a new one if there is none).
    exit : str, optional
        The member whose outlet a flowsheet receives. Default: the last
        member in evaluation order whose outlet is not sent entirely round a
        link.
    dae_integrator : IntegratorBackend, optional
        A DAE backend to fall back to if iteration fails and the configured
        integrator only solves ODEs.
    tear_tol : float, optional
        Relative tolerance on each tear field in iterate mode.
    max_tear_sweeps : int, optional
        Sweeps allowed per evaluation before iteration is declared failed.
    wegstein_bounds : (float, float), optional
        Bounds on the Wegstein acceleration factor. The lower bound sets how
        strong a recycle can be accelerated: a loop gain r needs
        q = r / (r - 1), so the default -100 covers r up to about 0.99.
    sync_members : bool, optional
        Bring every member to the evaluated state before any balance runs,
        so a controller can read another member. Default True; turning it off
        saves one phase update per member per evaluation when no member
        looks at another.
    """

    TEAR_MODES = ('iterate', 'algebraic')

    # Read off every unit by Connections and SimulationExec.
    material_from_upstream = False
    default_output = None
    bipartite = None
    names_upstream = None

    def __init__(self, members, links=(), integrator=None,
                 tear_mode='iterate', order=None, entry=None,
                 entry_inlet=None, exit=None, dae_integrator=None,
                 tear_tol=1e-12, max_tear_sweeps=50,
                 wegstein_bounds=(-100.0, 0.0), sync_members=True):

        if not isinstance(members, dict) or not members:
            raise TypeError(
                "members must be a non-empty dict of {name: vessel}.")

        owners = {}

        for name, member in members.items():

            if not isinstance(name, str) or not name.isidentifier():
                raise ValueError(
                    f"Member name {name!r} is not a valid identifier. Members "
                    "are reachable as attributes, so name them like "
                    "flowsheet units ('R01', 'CR01').")

            if not isinstance(member, MultiPhaseVessel):
                raise TypeError(
                    f"Member {name!r} is a {type(member).__name__}. Only "
                    "units built on MultiPhaseVessel can be integrated "
                    "concurrently; keep older unit operations in the "
                    "sequential part of the flowsheet.")

            if id(member) in owners:
                raise ValueError(
                    f"Members {owners[id(member)]!r} and {name!r} are the same "
                    "vessel object. Each member must be its own vessel.")

            owners[id(member)] = name

        if tear_mode not in self.TEAR_MODES:
            raise ValueError(
                f"tear_mode must be one of {self.TEAR_MODES}, got "
                f"{tear_mode!r}.")

        if int(max_tear_sweeps) < 1:
            raise ValueError("max_tear_sweeps must be at least 1.")

        if wegstein_bounds[0] > wegstein_bounds[1]:
            raise ValueError(
                f"wegstein_bounds must be (low, high), got {wegstein_bounds}.")

        self.members = dict(members)

        for role, name in (('entry', entry), ('exit', exit)):
            if name is not None and name not in self.members:
                raise KeyError(
                    f"{role} {name!r} is not a member. Members are "
                    f"{list(self.members)}.")

        self.integrator = ScipyBackend() if integrator is None else integrator
        self.dae_integrator = dae_integrator

        self.tear_mode = tear_mode

        # What the solves actually use. Starts as tear_mode and becomes
        # 'algebraic' for good if iteration ever fails.
        self.active_tear_mode = tear_mode

        # True when this system has to be integrated as a DAE. tear_fallback
        # records why, when it is because iteration failed.
        self.requires_dae = False
        self.tear_fallback = None

        self.tear_tol = float(tear_tol)
        self.max_tear_sweeps = int(max_tear_sweeps)
        self.wegstein_bounds = tuple(float(bound) for bound in wegstein_bounds)
        self.sync_members = bool(sync_members)

        self._requested_order = None if order is None else list(order)
        self._entry = entry
        self._entry_inlet = entry_inlet
        self._exit = exit

        # Runtime state, laid out by compile_structure.
        self.result = None
        self.derivatives = None
        self.solver_state_collection = None
        self.order = None
        self.tears = []
        self.state_event_list = []
        self.tear_statistics = {'evaluations': 0, 'sweeps': 0,
                                'max_sweeps': 0}

        self._member_slices = {}
        self._loop_start = 0
        self._tear_layouts = {}
        self._tear_slices = {}
        self._tear_blocks = {}
        self._tear_scales = None
        self._tear_residuals = {}
        self._tear_failure = None
        self._rate_buffer = None
        self._compiled_events = None
        self._event_owners = ()
        self._event_cache_key = None
        self._event_cache_values = None
        self._eval_key = None
        self._resume = None

        self.links = []

        for link in links:
            self.add_link(link)

        for name in self.members:
            if name in self.__dict__ or hasattr(type(self), name):
                raise ValueError(
                    f"Member name {name!r} collides with a ConcurrentVessel "
                    "attribute. Choose another name.")

    # =================================================================
    # Construction
    # =================================================================

    def add_link(self, link):
        """Connect two members. See StreamLink."""

        if not isinstance(link, StreamLink):
            link = StreamLink(*link)

        for role in ('source', 'sink'):
            name = getattr(link, role)
            if name not in self.members:
                raise KeyError(
                    f"Link {role} {name!r} is not a member. Members are "
                    f"{list(self.members)}.")

        if link.source == link.sink:
            raise ValueError(
                f"{link.source} cannot feed itself: a well-mixed vessel's "
                "outlet returned to its own inlet changes nothing.")

        if any(other.label == link.label for other in self.links):
            raise ValueError(
                f"A link labelled {link.label!r} already exists; give one of "
                "them a name.")

        if not 0.0 < link.fraction <= 1.0:
            raise ValueError(
                f"Link {link.label} fraction must be in (0, 1], got "
                f"{link.fraction}.")

        source = self.members[link.source]
        sink = self.members[link.sink]

        if source.oper_mode != 'Continuous' or not source.outlet_connections:
            raise TypeError(
                f"{link.source} is a {source.oper_mode or 'non-continuous'} "
                f"{type(source).__name__} with no outlet stream, so it cannot "
                f"feed {link.sink} while both run. Only a continuous unit "
                "discharges a stream.")

        if sink.oper_mode == 'Batch':
            raise TypeError(
                f"{link.sink} is a batch unit and cannot be fed directly from "
                f"{link.source}: a flow has to be accumulated into a charge "
                "first. Put a holding vessel between them.")

        # The sink receives a copy of the source's outlet stream, as
        # Connections hands a downstream unit a copy of the upstream Outlet.
        stream = copy.deepcopy(source.outlet_connections[0].stream)
        connection = sink.make_stream_connection(stream)

        coupling = CoupledInlet(source, link.fraction)

        for phase in connection.stream:
            phase.DynamicInlet = coupling

        connections = list(sink.inlet_connections)

        if link.inlet is None:
            connections.append(connection)
            link.inlet_index = len(connections) - 1
        else:
            if not 0 <= link.inlet < len(connections):
                raise IndexError(
                    f"{link.sink} has {len(connections)} inlet connection(s); "
                    f"link {link.label} cannot replace index {link.inlet}.")

            if link.inlet in self._coupled_inlet_indices(link.sink):
                raise ValueError(
                    f"{link.sink} inlet {link.inlet} is already fed by "
                    "another link.")

            connections[link.inlet] = connection
            link.inlet_index = link.inlet

        sink.inlet_connections = connections

        link.coupling = coupling
        self.links.append(link)

        self.solver_state_collection = None
        self._eval_key = None

        return link

    def _coupled_inlet_indices(self, member_name):
        return {link.inlet_index for link in self.links
                if link.sink == member_name}

    def _check_links(self):
        """Refuse a layout the links no longer describe."""

        for link in self.links:

            sink = self.members[link.sink]
            connections = sink.inlet_connections

            attached = (
                link.inlet_index < len(connections)
                and any(getattr(phase, 'DynamicInlet', None) is link.coupling
                        for phase in connections[link.inlet_index].stream)
            )

            if not attached:
                raise RuntimeError(
                    f"{link.sink}'s inlet connections were replaced after it "
                    f"was linked to {link.source}, so link {link.label} no "
                    "longer reaches it. Configure each member's own feeds "
                    "before building the ConcurrentVessel.")

        for name in self.members:
            sent = 1.0 - self.product_fraction_of(name)

            if sent > 1.0 + 1e-12:
                raise ValueError(
                    f"Links take {sent:.6g} of {name}'s outlet; at most all "
                    "of it can be sent on.")

    def product_fraction_of(self, member_name):
        """Share of a member's outlet that no link takes."""

        return 1.0 - sum(link.fraction for link in self.links
                         if link.source == member_name)

    # =================================================================
    # Structure
    # =================================================================

    def _evaluation_order(self):

        names = list(self.members)

        if self._requested_order is not None:
            order = list(self._requested_order)

            if sorted(order) != sorted(names):
                raise ValueError(
                    f"order must name every member exactly once: got {order}, "
                    f"members are {names}.")

            return order

        successors = {name: [] for name in names}

        for link in self.links:
            successors[link.source].append(link.sink)

        visited = set()
        postorder = []

        def visit(name):
            visited.add(name)

            for successor in successors[name]:
                if successor not in visited:
                    visit(successor)

            postorder.append(name)

        for name in names:
            if name not in visited:
                visit(name)

        # Reverse postorder: upstream first for an acyclic system, and in a
        # loop every link pointing back to an earlier member is a DFS back
        # edge -- the fewest tears this visiting order allows.
        return postorder[::-1]

    def compile_structure(self):
        """
        Compile every member, then lay out the combined system.

        Like MultiPhaseVessel.compile_structure, this runs at the top of
        every solve.
        """

        times = [float(member.elapsed_time) for member in self.members.values()]

        if max(times) - min(times) > 1e-9 * max(1.0, abs(max(times))):
            raise ValueError(
                "Members are at different times "
                f"({dict(zip(self.members, times))}). A ConcurrentVessel "
                "advances its members together, so they must start together.")

        self._check_links()

        for member in self.members.values():
            member.compile_structure()

        self.order = self._evaluation_order()

        position = {name: index for index, name in enumerate(self.order)}

        for link in self.links:
            link.is_tear = position[link.source] >= position[link.sink]

        self.tears = [link for link in self.links if link.is_tear]

        # Members ahead of every tear's sink cannot depend on a tear, so a
        # repeat sweep starts here.
        self._loop_start = min(
            (position[link.sink] for link in self.tears),
            default=len(self.order),
        )

        self._tear_layouts = {
            link.label: self._tear_layout(link) for link in self.tears
        }

        self._tear_scales = None
        self._tear_failure = None

        self.tear_statistics = {'evaluations': 0, 'sweeps': 0,
                                'max_sweeps': 0}

        self._compile_layout()
        self.compile_events()

        self.requires_dae = self.has_algebraic_balance

        self._eval_key = None

    def _tear_layout(self, link):
        """(field, dim) for each field the source's outlet carries."""

        source = self.members[link.source]
        layout = []

        for field, key in source._outlet_fields().items():

            variable = (source.solver_state_collection.states.get(key)
                        or source.output_state_collection.states.get(key))

            if variable is not None:
                layout.append((field, int(variable.dim)))

        return layout

    def _compile_layout(self):
        """
        One state collection over every member, plus any algebraic tears.

        Member blocks are copies of each member's own layout, in evaluation
        order and with the member's name prefixed, so the combined vector is
        the members' vectors laid end to end.
        """

        collection = StateCollection()
        start = 0

        self._member_slices = {}

        for name in self.order:

            member_collection = self.members[name].solver_state_collection

            for key, state in member_collection.states.items():
                variable = copy.copy(state)
                variable.name = f"{name}.{state.name}"
                variable.phaseref = key.phaseref
                collection.add(variable, error_on_conflict=True)

            dim = member_collection.dim
            self._member_slices[name] = slice(start, start + dim)
            start += dim

        self._tear_slices = {}
        self._tear_blocks = {}

        if self.active_tear_mode == 'algebraic':

            for link in self.tears:

                block_start = start
                slices = {}

                for field, dim in self._tear_layouts[link.label]:
                    collection.add(
                        StateVariable(
                            name=f"tear[{link.label}].{field}",
                            dim=dim,
                            units="scaled",
                            state_type=StateCollection.ALGEBRAIC_STATE_TYPE,
                            limit_negative_inventory=False,
                        ),
                        error_on_conflict=True,
                    )

                    slices[field] = slice(start, start + dim)
                    start += dim

                self._tear_slices[link.label] = slices
                self._tear_blocks[link.label] = slice(block_start, start)

        collection.compile()

        if collection.dim != start:
            raise RuntimeError(
                f"Combined layout has {collection.dim} states but the member "
                f"and tear blocks add up to {start}.")

        self.solver_state_collection = collection
        self._rate_buffer = np.zeros(start)

    @property
    def has_algebraic_balance(self):
        if self.solver_state_collection is None:
            return False

        return self.solver_state_collection.has_algebraic

    @property
    def concurrent_members(self):
        """The members, under the name CheckModule looks for."""
        return self.members

    @property
    def elapsed_time(self):
        return next(iter(self.members.values())).elapsed_time

    @elapsed_time.setter
    def elapsed_time(self, value):
        for member in self.members.values():
            member.elapsed_time = value

    def __getitem__(self, name):
        return self.members[name]

    def __getattr__(self, name):
        # Only reached for a name that is not a real attribute.
        members = self.__dict__.get('members')

        if members is not None and name in members:
            return members[name]

        # A property whose getter raises AttributeError looks to Python like
        # a missing attribute, which would bury the getter's explanation
        # under a bare name. Re-run it so its own message gets through.
        descriptor = getattr(type(self), name, None)

        if isinstance(descriptor, property) and descriptor.fget is not None:
            return descriptor.fget(self)

        raise AttributeError(
            f"{type(self).__name__} has no attribute or member {name!r}")

    def __repr__(self):
        members = ', '.join(f"{name} ({type(member).__name__})"
                            for name, member in self.members.items())
        links = ', '.join(repr(link) for link in self.links)

        return (f"ConcurrentVessel(members=[{members}], links=[{links}], "
                f"tear_mode={self.active_tear_mode!r})")

    # =================================================================
    # Integrator protocol
    # =================================================================

    def reset(self):

        for member in self.members.values():
            member.reset()

        # A solve restarted part-way (the fallback to algebraic tears) resumes
        # from where it began, not from the members' initial charge.
        if self._resume is not None:
            for name, (elapsed, state) in self._resume.items():
                member = self.members[name]
                member.update_phases_from_state(
                    member.complete_state(dict(state), elapsed))
                member.elapsed_time = elapsed

        for link in self.links:
            link.coupling.override = None
            link.coupling.frozen = None
            link.coupling.last_served = None

        self.result = None
        self._eval_key = None
        self._event_cache_key = None

    def _snapshot(self):
        """Every member's state now, for resuming a restarted solve."""

        snapshot = {}

        for name, member in self.members.items():
            state = member.solver_state_collection.unpack(
                member.create_solver_init_states())

            snapshot[name] = (
                member.elapsed_time,
                {key: np.array(value, dtype=float) for key, value in state.items()},
            )

        return snapshot

    def configure_solver(self):
        """
        Pass the integrator the strongest linear-solver request among the
        members: one large sparse member makes the whole system large and
        sparse.
        """

        # A member's hint describes its own ODE. Algebraic tear states make a
        # different Newton system, one an unpreconditioned Krylov solve
        # handles badly: on a crystallizer recycle, IDA with GMRES had
        # reached t=1.7e-4 s after 72000 right-hand sides, where a dense
        # solve covered 300 s in 40 s. So those systems are solved densely.
        if self._tear_slices:
            self.integrator.set_linear_solver('dense')
            return

        hints = {member.linear_solver_hint
                 for member in self.members.values()} - {None}

        for kind in ('krylov', 'sparse', 'dense'):
            if kind in hints:
                self.integrator.set_linear_solver(kind)
                return

    def create_solver_init_states(self):

        parts = [
            np.asarray(self.members[name].create_solver_init_states(),
                       dtype=float)
            for name in self.order
        ]

        if self._tear_slices:
            parts.append(self._initial_tear_states(np.concatenate(parts)))

        return np.concatenate(parts)

    def save_initial_solver_state(self, states=None, time=None):

        if states is None:
            states = self.create_solver_init_states()

        if time is None:
            time = self.elapsed_time

        for name in self.order:
            self.members[name].save_initial_solver_state(
                states[self._member_slices[name]], time)

    def unit_model(self, time, states, params=None, sw=None):

        states = np.asarray(states, dtype=float)

        self._evaluate(time, states)

        rates = self._rate_buffer

        for name in self.order:
            rates = np.set_at(rates, self._member_slices[name],
                              self.members[name].derivatives)

        for label, block in self._tear_blocks.items():
            rates = np.set_at(rates, block, self._tear_residuals[label])

        self.derivatives = rates

        return rates

    # =================================================================
    # Evaluation
    # =================================================================

    def _split(self, states):
        return {name: states[self._member_slices[name]] for name in self.order}

    def _sync(self, time, member_states):
        """Bring every member's phases to the state being evaluated."""

        for name in self.order:
            member = self.members[name]
            completed = member.complete_state(
                member.solver_state_collection.unpack(member_states[name]),
                time)
            member.update_phases_from_state(completed)

    def _sweep(self, time, member_states, start=0):
        """Evaluate members in order, each reading those already evaluated."""

        for name in self.order[start:]:
            member = self.members[name]
            member.invalidate_inlet_cache()
            member.unit_model(time=time, states=member_states[name])

    def _evaluate(self, time, states):
        """
        Bring every member, and every coupling, to (time, states).

        Memoized on the exact point: the backends commonly ask for the
        right-hand side and the events at the same state, and the members
        already hold it.
        """

        states = np.asarray(states, dtype=float)

        # A traced point has no bytes to memoize on, and is evaluated once.
        traced = np.is_traced((time, states))
        key = None if traced else (float(time), states.tobytes())

        if not traced and key == self._eval_key:
            return

        if traced and self.tears and not self._tear_slices:
            raise NotImplementedError(
                f"Recycle {', '.join(link.label for link in self.tears)} is "
                "converged by iterating inside the right-hand side, which a "
                "trace cannot follow: how many sweeps it takes depends on "
                "the values. Use tear_mode='algebraic', which carries the "
                "tear streams as algebraic states instead, with a backend "
                "that solves differential-algebraic systems.")

        self._eval_key = None

        member_states = self._split(states)

        if self.sync_members:
            self._sync(time, member_states)

        if self._tear_slices:

            for link in self.tears:
                link.coupling.override = self._unpack_tear(link, states)

            self._sweep(time, member_states)

            for link in self.tears:
                computed = self._pack_tear(link, link.coupling.live_values())
                self._tear_residuals[link.label] = (
                    states[self._tear_blocks[link.label]] - computed)

        elif self.tears:
            self._iterate(time, member_states)

        else:
            self._sweep(time, member_states)

        self._eval_key = key

    def _iterate(self, time, member_states, raise_on_failure=True):
        """
        Converge every tear by substitution, accelerated with Wegstein.

        Each tear starts from the value it converged to at the previous
        evaluation, which is close, so a few sweeps are usually enough.
        """

        low, high = self.wegstein_bounds

        start = 0

        for link in self.tears:
            if link.coupling.override is None:
                link.coupling.override = self._priming_values(link)

        # A tear seeded without its source's outlet has nothing to be
        # compared against, so one sweep establishes it and does not count
        # towards max_tear_sweeps.
        if any(not self._tear_complete(link) for link in self.tears):

            self._sweep(time, member_states)
            start = self._loop_start

            for link in self.tears:
                link.coupling.override = link.coupling.live_values()

        history = {}
        worst = np.inf

        for sweep in range(1, self.max_tear_sweeps + 1):

            self._sweep(time, member_states, start)
            start = self._loop_start

            worst = 0.0

            for link in self.tears:
                worst = max(worst,
                            self._update_tear(link, history, low, high))

            if worst <= self.tear_tol:
                break

        else:
            return self._tear_iteration_failed(
                time, worst, sweep, raise_on_failure)

        self._tear_failure = None

        stats = self.tear_statistics
        stats['evaluations'] += 1
        stats['sweeps'] += sweep
        stats['max_sweeps'] = max(stats['max_sweeps'], sweep)

        return sweep

    def _tear_complete(self, link):
        override = link.coupling.override

        return override is not None and all(
            field in override for field, _ in self._tear_layouts[link.label])

    def _tear_iteration_failed(self, time, worst, sweeps, raise_on_failure):

        failure = {
            'time': float(time),
            'residual': float(worst),
            'sweeps': sweeps,
            'links': [link.label for link in self.tears],
        }

        self._tear_failure = failure

        if raise_on_failure:
            raise TearConvergenceError(
                f"Recycle {', '.join(failure['links'])} did not converge "
                f"at t={time:g}: tear residual {worst:.2e} (tolerance "
                f"{self.tear_tol:.0e}) after {sweeps} sweeps.")

        return sweeps

    def _priming_values(self, link):
        """
        A first guess for a tear whose source has not been evaluated yet.

        No flow: the sink starts from its template composition with nothing
        arriving, and the first sweep replaces the guess with what the
        source actually discharges.
        """

        try:
            return link.coupling.live_values()
        except RuntimeError:
            return {'vol_flow': 0.0}

    def _update_tear(self, link, history, low, high):
        """
        One substitution step on a tear; returns its relative residual.

        Each field is measured against its own magnitude, so a flow of 1e-5
        m3/s is held to the same relative standard as a temperature of 300 K.
        """

        coupling = link.coupling
        computed = coupling.live_values()
        guess = coupling.override
        layout = self._tear_layouts[link.label]

        if guess is None or any(field not in guess for field, _ in layout):
            coupling.override = computed
            history.pop(link.label, None)
            return np.inf

        error = 0.0
        x_parts, g_parts, scale_parts = [], [], []

        for field, dim in layout:
            x = np.atleast_1d(np.asarray(guess[field], dtype=float))
            g = np.atleast_1d(np.asarray(computed[field], dtype=float))

            scale = max(float(np.max(np.abs(x))), float(np.max(np.abs(g))),
                        np.finfo(float).tiny)

            error = max(error, float(np.max(np.abs(g - x))) / scale)

            x_parts.append(x)
            g_parts.append(g)
            scale_parts.append(np.full(dim, scale))

        if error <= self.tear_tol:
            # The freshest value warm-starts the next evaluation.
            coupling.override = computed
            history.pop(link.label, None)
            return error

        x = np.concatenate(x_parts)
        g = np.concatenate(g_parts)
        scale = np.concatenate(scale_parts)

        new = g.copy()
        previous = history.get(link.label)

        if previous is not None:
            x_old, g_old = previous
            dx = x - x_old

            # Below this the secant slope is round-off, and accelerating it
            # would amplify noise rather than converge.
            active = np.abs(dx) > 1e-8 * scale

            if active.any():
                slope = (g[active] - g_old[active]) / dx[active]

                with np.errstate(divide='ignore', invalid='ignore'):
                    q = np.where(np.abs(slope - 1.0) > 1e-12,
                                 slope / (slope - 1.0), 0.0)

                q = np.clip(q, low, high)
                new[active] = q * x[active] + (1.0 - q) * g[active]

        history[link.label] = (x, g)
        coupling.override = self._fields_from_vector(link, new)

        return error

    def _fields_from_vector(self, link, vector):
        values = {}
        position = 0

        for field, dim in self._tear_layouts[link.label]:
            block = vector[position:position + dim]
            values[field] = np.as_float(block[0]) if dim == 1 else block.copy()
            position += dim

        return values

    # ----- algebraic tears ----------------------------------------------

    def _pack_tear(self, link, values):
        """A tear's fields as its scaled block of the solver vector."""

        scales = self._tear_scales[link.label]
        parts = []

        for field, dim in self._tear_layouts[link.label]:

            if field not in values:
                raise KeyError(
                    f"Link {link.label} carried no {field!r}; its source "
                    f"publishes {sorted(values)}.")

            value = np.atleast_1d(np.asarray(values[field], dtype=float))

            if value.size != dim:
                raise ValueError(
                    f"Link {link.label} field {field!r} has {value.size} "
                    f"values; the layout expects {dim}.")

            parts.append(value / scales[field])

        return np.concatenate(parts)

    def _unpack_tear(self, link, states):
        """A tear's block of the solver vector, back in physical units."""

        scales = self._tear_scales[link.label]
        values = {}

        for field, dim in self._tear_layouts[link.label]:
            block = states[self._tear_slices[link.label][field]] * scales[field]
            values[field] = np.as_float(block[0]) if dim == 1 else block.copy()

        return values

    def _initial_tear_states(self, member_vector):
        """
        Starting values for the algebraic tears, scaled.

        Found by iterating at the starting point, as far as iteration gets;
        a DAE backend makes them consistent before it takes a step.
        """

        time = self.elapsed_time
        member_states = self._split(member_vector)

        if self.sync_members:
            self._sync(time, member_states)

        self._iterate(time, member_states, raise_on_failure=False)
        self._eval_key = None

        values = {link.label: link.coupling.override for link in self.tears}

        if self._tear_scales is None:
            self._tear_scales = self._choose_tear_scales(values)

        return np.concatenate([
            self._pack_tear(link, values[link.label]) for link in self.tears
        ])

    def _choose_tear_scales(self, values):
        """
        One scale per field, so every algebraic tear state is O(1).

        Integrators judge convergence with absolute tolerances too, and an
        unscaled flow of 1e-5 m3/s would sit below a default atol of 1e-6.
        A field that starts at zero borrows the magnitude of the same field
        on another link, or 1.
        """

        magnitudes = {}

        for link in self.tears:
            for field, _ in self._tear_layouts[link.label]:
                magnitude = float(np.max(np.abs(np.atleast_1d(
                    values[link.label][field]))))
                magnitudes.setdefault(field, []).append(magnitude)

        scales = {}

        for link in self.tears:
            scales[link.label] = {}

            for field, _ in self._tear_layouts[link.label]:
                magnitude = float(np.max(np.abs(np.atleast_1d(
                    values[link.label][field]))))

                if magnitude <= 0.0:
                    magnitude = max(magnitudes[field])

                scales[link.label][field] = magnitude if magnitude > 0.0 else 1.0

        return scales

    @contextmanager
    def _frozen_couplings(self):
        """Pin every coupling at the value the last evaluation served."""

        for link in self.links:
            link.coupling.frozen = link.coupling.last_served

        try:
            yield
        finally:
            for link in self.links:
                link.coupling.frozen = None

    # =================================================================
    # Events
    # =================================================================

    def compile_events(self):
        """Every member's switching surfaces, in evaluation order."""

        events = []
        owners = []

        for name in self.order:
            for index, event in enumerate(self.members[name].compiled_events):
                events.append(event)
                owners.append((name, index))

        self._compiled_events = tuple(events)
        self._event_owners = tuple(owners)
        self._event_cache_key = None
        self._event_cache_values = None

        return self._compiled_events

    @property
    def compiled_events(self):
        if self._compiled_events is None:
            self.compile_events()

        return self._compiled_events

    @property
    def has_events(self):
        return len(self.compiled_events) > 0

    def evaluate_events(self, time, states):
        """
        Evaluate every member's events at (time, states).

        The whole system is evaluated first, because an event function may
        resolve its vessel's inlets, and a coupled inlet is only valid once
        the member feeding it has been evaluated at this point.
        """

        events = self.compiled_events

        if not events:
            return np.zeros(0)

        states = np.asarray(states, dtype=float)
        cache_key = (float(time), states.tobytes())

        if self._event_cache_key == cache_key:
            return self._event_cache_values

        self._evaluate(time, states)

        values = np.empty(len(events))
        position = 0

        with self._frozen_couplings():
            for name in self.order:

                member = self.members[name]
                count = len(member.compiled_events)

                if not count:
                    continue

                member._event_cache_key = None
                member.invalidate_inlet_cache()

                values[position:position + count] = member.evaluate_events(
                    time, states[self._member_slices[name]])

                position += count

        self._event_cache_key = cache_key
        self._event_cache_values = values

        return values

    def handle_event(self, time, event_indices):
        """Hand each located event to its member; True stops the run."""

        stop = False

        for index in event_indices:
            name, local = self._event_owners[index]
            stop = bool(self.members[name].handle_event(time, [local])) or stop

        return stop

    # =================================================================
    # Solve
    # =================================================================

    def _dae_capable_integrator(self):
        for candidate in (self.integrator, self.dae_integrator):
            if candidate is not None and candidate.supports_algebraic:
                return candidate

        return None

    def solve_unit(self, runtime=None, time_grid=None, **kwargs):

        self.compile_structure()

        if self.has_algebraic_balance:
            # Prefer a DAE backend the caller supplied over refusing; with
            # none, the configured backend refuses and says why.
            capable = self._dae_capable_integrator()

            if capable is not None:
                self.integrator = capable

        resume = self._snapshot() if self.elapsed_time > 0 else None

        try:
            return self.integrator.solve(
                self, runtime=runtime, time_grid=time_grid, **kwargs)

        except Exception as error:

            failure = self._tear_failure

            if (failure is None or self.active_tear_mode != 'iterate'
                    or not self.tears):
                raise

            return self._fall_back_to_algebraic(
                failure, error, resume, runtime, time_grid, kwargs)

    def _fall_back_to_algebraic(self, failure, error, resume, runtime,
                                time_grid, kwargs):
        """Restart this solve with the tears as algebraic states."""

        self.tear_fallback = dict(failure)
        self.active_tear_mode = 'algebraic'
        self.requires_dae = True

        labels = ', '.join(failure['links'])
        integrator = self._dae_capable_integrator()

        if integrator is None:
            raise TearConvergenceError(
                f"Recycle {labels} did not converge by iteration inside the "
                f"right-hand side at t={failure['time']:g} (tear residual "
                f"{failure['residual']:.2e} after {failure['sweeps']} "
                "sweeps), so it has to be solved with the recycle as "
                "algebraic states, and "
                f"{type(self.integrator).__name__} integrates ODEs only. Use "
                "a DAE backend -- integrator=AssimuloDAEBackend() or "
                "DiffeqpyBackend() -- or pass dae_integrator=... to have this "
                "fallback happen automatically. requires_dae is now set."
            ) from error

        warnings.warn(
            f"Recycle {labels} did not converge by iteration at "
            f"t={failure['time']:g} (tear residual {failure['residual']:.2e} "
            f"after {failure['sweeps']} sweeps). Restarting this solve with "
            f"the recycle as algebraic states on {type(integrator).__name__}.",
            RuntimeWarning,
            stacklevel=3,
        )

        self.integrator = integrator
        self._resume = resume

        try:
            self.compile_structure()
            integrator.invalidate()

            return integrator.solve(
                self, runtime=runtime, time_grid=time_grid, **kwargs)
        finally:
            self._resume = None

    # =================================================================
    # Results: the same steps as MultiPhaseVessel.retrieve_results, run
    # for every member together
    # =================================================================

    def retrieve_results(self, time, solver_states):

        time = np.asarray(time, dtype=float).reshape(-1)
        solver_states = np.asarray(solver_states, dtype=float).reshape(
            time.size, -1)

        solver_history = self.build_solver_history(time, solver_states)

        output_history, coupling_history = (
            self.find_output_states_from_replay(time, solver_states))

        completed_state = self.update_final_state(solver_history)

        self.update_final_conditions(
            completed_state, time, solver_states, output_history)

        self.result = self.build_dynamic_result(
            time, solver_history, output_history, coupling_history)

        return self.result

    def retrieve_sensitivities(self, time, solver_states, sensitivities,
                               parameters, output_tangents=None):
        """
        Each member's solver-state sensitivities, on its own result.

        As MultiPhaseVessel.retrieve_sensitivities, for the members' solver
        states only: an output state of one member depends on the others
        through the couplings, and the joint replay that computes it is not
        differentiated yet. ``output_tangents`` is ignored.
        """

        time = np.asarray(time, dtype=float).reshape(-1)
        sensitivities = np.asarray(sensitivities, dtype=float).reshape(
            time.size, np.shape(solver_states)[-1], -1)

        fields = {}

        for name in self.order:
            member = self.members[name]
            collection = member.solver_state_collection
            block = sensitivities[:, self._member_slices[name], :]

            member_fields = {}

            for key, state_slice, state in zip(collection.keys,
                                               collection.slices.values(),
                                               collection.state_values):
                values = block[:, state_slice, :]

                if state.dim == 1:
                    values = values[:, 0, :]

                member_fields[collection.format_key(key)] = values

            member.result.sensitivities = member_fields
            member.result.param_names = parameters.free_names
            fields[name] = member_fields

        self.result.sensitivities = fields
        self.result.param_names = parameters.free_names

        return fields

    def build_solver_history(self, time, solver_states):
        return {
            name: self.members[name].build_solver_history(
                time, solver_states[:, self._member_slices[name]])
            for name in self.order
        }

    def create_pseudo(self, memo=None):
        """
        Independent copies of every member, for the joint replay.

        The members are copied through one shared memo, with an empty
        pseudo registered for each before anything is copied. A reference
        from one member to another -- a coupling's source, a controller
        watching a neighbour -- therefore lands on that neighbour's pseudo,
        which the replay keeps at the right state, rather than on a frozen
        copy of the real vessel taken at the end of the run.

        ``memo``, when given, is filled in, so a caller can find the copy of
        any object inside (PharmaPy.Parameters.bind does).
        """

        if memo is None:
            memo = {}

        for member in self.members.values():
            memo.setdefault(id(member), object.__new__(type(member)))

        members = {name: member.create_pseudo(memo)
                   for name, member in self.members.items()}

        pseudo = object.__new__(type(self))
        memo[id(self)] = pseudo

        excluded = {'integrator', 'dae_integrator', 'result', 'derivatives',
                    'members'}

        for name, value in self.__dict__.items():
            if name in excluded:
                continue

            pseudo.__dict__[name] = copy.deepcopy(value, memo)

        pseudo.members = members
        pseudo.integrator = None
        pseudo.dae_integrator = None
        pseudo.result = None
        pseudo.derivatives = None
        pseudo._eval_key = None

        return pseudo

    def find_output_states_from_replay(self, time, solver_states):
        """
        Replay the accepted trajectory with every member together.

        At each point the whole system is evaluated first, so each coupled
        inlet carries what it carried during the run and every member is at
        the point's state. Each member's output states are then computed by
        its own MultiPhaseVessel.output_states_at -- the step a standalone
        replay takes -- with the couplings pinned.

        Returns ({member: {StateKey: array}}, {link label: {field: array}}).
        """

        pseudo = self.create_pseudo()

        # Start the tears where the integration started them, rather than
        # at the values they converged to at the end of the run.
        for link in pseudo.links:
            link.coupling.override = None

        output_history = {
            name: {key: [] for key in member.output_state_collection.states}
            for name, member in pseudo.members.items()
        }

        coupling_rows = {link.label: [] for link in pseudo.links}

        for index, t in enumerate(time):

            states = solver_states[index]

            pseudo._evaluate(t, states)

            for link in pseudo.links:
                coupling_rows[link.label].append(link.coupling.last_served)

            with pseudo._frozen_couplings():
                for name in pseudo.order:

                    member = pseudo.members[name]
                    member.invalidate_inlet_cache()

                    solver_state = member.solver_state_collection.unpack(
                        states[pseudo._member_slices[name]])

                    outputs = member.output_states_at(t, solver_state)

                    for key, value in outputs.items():
                        output_history[name][key].append(value)

        for history in output_history.values():
            for key in history:
                history[key] = np.asarray(history[key])

        coupling_history = {
            label: self._stack_rows(rows)
            for label, rows in coupling_rows.items()
        }

        return output_history, coupling_history

    @staticmethod
    def _stack_rows(rows):
        """[{field: value}] per time point -> {field: array over time}."""

        fields = []

        for row in rows:
            for field in row or {}:
                if field not in fields:
                    fields.append(field)

        stacked = {}

        for field in fields:
            sample = next(np.asarray(row[field], dtype=float)
                          for row in rows if row and field in row)

            stacked[field] = np.array([
                np.asarray(row[field], dtype=float)
                if row and field in row else np.full(sample.shape, np.nan)
                for row in rows
            ])

        return stacked

    def update_final_state(self, solver_history):
        return {
            name: self.members[name].update_final_state(solver_history[name])
            for name in self.order
        }

    def update_final_conditions(self, completed_state, time, solver_states,
                                output_history):
        """
        Each member's final conditions, with the whole system at the final
        point and every coupling pinned at what it carried there.
        """

        self._eval_key = None
        self._evaluate(time[-1], solver_states[-1])

        with self._frozen_couplings():
            for name in self.order:
                member = self.members[name]
                member.invalidate_inlet_cache()
                member.update_final_conditions(
                    completed_state[name], time, None, output_history[name])

        self._eval_key = None
        self._event_cache_key = None

    def build_dynamic_result(self, time, solver_history, output_history,
                             coupling_history):

        results = {}

        for name in self.order:
            member = self.members[name]
            member.result = member.build_dynamic_result(
                time, solver_history[name], output_history[name])

        for name, member in self.members.items():
            results[name] = member.result

        return ConcurrentResult(time, results, coupling_history)

    # =================================================================
    # Flowsheet node: what SimulationExec and Connections read
    # =================================================================

    @property
    def entry(self):
        """The member a flowsheet feed arrives through."""

        if self._entry is not None:
            return self._entry

        return next(iter(self.members))

    @property
    def exit(self):
        """The member whose outlet a flowsheet receives, or None."""

        if self._exit is not None:
            return self._exit

        order = self.order if self.order is not None else self._evaluation_order()

        for name in reversed(order):
            if self.product_fraction_of(name) > 1e-12:
                return name

        return None

    @property
    def product_fraction(self):
        """Share of the exit member's outlet that leaves the system."""
        return self.product_fraction_of(self._exit_name())

    def _exit_name(self):
        name = self.exit

        if name is None:
            raise ValueError(
                "Every member's outlet is sent round a link, so nothing "
                "leaves this ConcurrentVessel for a downstream unit to "
                "receive.")

        return name

    @property
    def oper_mode(self):
        modes = {member.oper_mode for member in self.members.values()}

        if len(modes) == 1:
            return modes.pop()

        return self.outlet_oper_mode

    @property
    def inlet_oper_mode(self):
        return self.members[self.entry].oper_mode

    @property
    def outlet_oper_mode(self):
        return self.members[self._exit_name()].oper_mode

    @property
    def is_continuous(self):
        return self.outlet_oper_mode == 'Continuous'

    @property
    def names_states_in(self):
        return self.members[self.entry].names_states_in

    @property
    def names_states_out(self):
        return self.members[self._exit_name()].names_states_out

    def _entry_inlet_index(self):
        member = self.members[self.entry]
        coupled = self._coupled_inlet_indices(self.entry)
        count = len(member.inlet_connections)

        if self._entry_inlet is not None:

            if self._entry_inlet in coupled:
                raise ValueError(
                    f"entry_inlet {self._entry_inlet} of {self.entry} is fed "
                    "by a link; a flowsheet feed needs an inlet of its own.")

            if not 0 <= self._entry_inlet <= count:
                raise IndexError(
                    f"{self.entry} has {count} inlet connection(s); "
                    f"entry_inlet {self._entry_inlet} is out of range.")

            return self._entry_inlet

        for index in range(count):
            if index not in coupled:
                return index

        return count

    @property
    def Inlet(self):
        raise AttributeError(
            "Inlet is a convenience API for feeding the entry member; read "
            "inlet_connections on the member instead.")

    @Inlet.setter
    def Inlet(self, inlet):
        """Feed the entry member, leaving its linked inlets in place."""

        streams = inlet if isinstance(inlet, (list, tuple)) else [inlet]

        if len(streams) != 1:
            raise ValueError(
                "A ConcurrentVessel takes one flowsheet feed, through its "
                f"entry member; got {len(streams)}.")

        member = self.members[self.entry]
        index = self._entry_inlet_index()

        connections = list(member.inlet_connections)
        connection = member.make_stream_connection(streams[0])

        if index == len(connections):
            connections.append(connection)
        else:
            connections[index] = connection

        member.inlet_connections = connections

        self._eval_key = None

    @property
    def Phases(self):
        """The entry member's phases, which a batch charge arrives into."""
        return self.members[self.entry].Phases

    @Phases.setter
    def Phases(self, phases):
        self.members[self.entry].Phases = phases
        self._eval_key = None

    @property
    def Outlet(self):
        """The exit member's outlet, less what the links send back."""

        outlet = self.members[self._exit_name()].Outlet
        share = self.product_fraction

        if share == 1.0:
            return outlet

        outlet = copy.deepcopy(outlet)
        phases = outlet if hasattr(outlet, 'Phases') else [outlet]

        for phase in phases:
            phase.updatePhase(vol_flow=phase.vol_flow * share)

        return outlet

    @property
    def outputs(self):
        """The exit member's outlet trajectory, less what the links take."""

        outputs = self.members[self._exit_name()].outputs

        if outputs is None:
            return None

        share = self.product_fraction

        if share != 1.0 and 'vol_flow' in outputs:
            outputs = dict(outputs)
            outputs['vol_flow'] = outputs['vol_flow'] * share

        return outputs

    @property
    def states_di(self):
        if self.solver_state_collection is None:
            return None

        return {
            StateCollection.format_key(key): state.as_dict()
            for key, state in self.solver_state_collection.states.items()
        }

    def flatten_states(self):
        """No-op kept for SimulationExec, as on MultiPhaseVessel."""
        return self.result
