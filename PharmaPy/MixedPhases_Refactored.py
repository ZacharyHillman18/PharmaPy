# -*- coding: utf-8 -*-
"""
Multi-phase containers for the MultiPhaseVessel refactor.

Carved out of MixedPhases.py when the phase stack was split. The legacy
Slurry / SlurryStream / Cake classes stay in PharmaPy.MixedPhases, which the
old unit operations still import; nothing here depends on them.
"""

from PharmaPy.Phases_Refactored import BasePhase
from collections import defaultdict
from PharmaPy import xp as np
import copy

class MixedPhase:
    # -----------------------------------------------------------------
    # Legacy connection protocol.
    #
    # PharmaPy.Connections reads these off whatever matter it is moving
    # between units, and sets them when it wires a connection. They are
    # declared on the class so __getattr__ -- which aggregates across
    # phases -- never sees them; none of them is a per-phase quantity.
    # Remove once the remaining unit operations are ported.
    # -----------------------------------------------------------------
    DynamicInlet = None
    y_upstream = None
    y_inlet = None
    time_upstream = None
    transferred_from_uo = False

    # Not a per-phase quantity either: without this, MixedPhase.ind_solv would
    # fall through to __getattr__ and come back as a mass-weighted *average of
    # species indices*, which is meaningless and would not look wrong.
    ind_solv = None
    name_solv = None

    def __init__(self,phases=None):
        self._Phases = []
        if phases is not None:
            self.Phases = phases
        self.EXTENSIVE_PROPERTIES = {"mass", "vol", "moles"}
    
    @property
    def Phases(self):
        return self._Phases
    
    @Phases.setter
    def Phases(self,phases):

        phases = phases if isinstance(phases,(list,tuple)) else [phases]
        phases = self._normalize_phases(phases)
        self._Phases = phases

    def _normalize_phases(self,phases):
        adjusted_phases = []

        for phase in phases:

            if isinstance(phase, MixedPhase):
                adjusted_phases.extend(self._normalize_phases(phase.Phases))

            elif isinstance(phase, BasePhase) and not phase.is_stream:
                adjusted_phases.append(phase)

            else:
                raise TypeError("All phases must be PharmaPy Phase objects")
        return adjusted_phases

    @property
    def phases_by_type(self):
        groups = defaultdict(list)
        for phase in self.Phases:
            key=phase.phase_family
            groups[key].append(phase)
        return groups
    def get_phase_from_ref(self,
            phase_ref
        )->BasePhase:

        candidates = []

        for phase in self.Phases:

            phase_type = (
                phase.phase_family
                .lower()
            )

            if phase_type == phase_ref.phase_type:
                candidates.append(phase)

        return candidates[phase_ref.index]
    def __iter__(self):
        return iter(self._Phases)
    
    def __len__(self):
        return len(self._Phases)

    def __getitem__(self, idx):
        return self._Phases[idx]
    
    def __getattr__(self, name:str):
        """
        Default bulk-property implementation.

        If a property or method is explicitly implemented on MixedPhase,
        Python will find it before calling __getattr__. This method therefore
        only handles attributes that are not otherwise defined.
        """

        # Prevent interfering with Python internals
        if (name.startswith("__") and name.endswith("__")) or name.startswith("_"):
            raise AttributeError(name)

        key =name.removesuffix("s").lower()
        if key in self.phases_by_type:
            out = MixedPhase(self.phases_by_type[key])
            return out
        phases = object.__getattribute__(self,"_Phases")
        masses = np.array([phase.mass for phase in phases], dtype=float) 

        total_mass = masses.sum()
        positive = total_mass > 0
        weights = np.where(positive, masses / np.where(positive, total_mass, 1.0),
                           np.ones(len(phases), dtype=float) / len(phases))

        attrs = [getattr(phase, name, None) for phase in phases]

        # Methods
        if any(callable(attr) for attr in attrs):

            def wrapper(*args, **kwargs):
                values = []

                for attr in attrs:
                    if attr is None:
                        values.append(0)
                    else:
                        values.append(attr(*args, **kwargs))

                values = np.asarray(values)

                # np.dot contracts the last axis of weights with the
                # SECOND-to-last of values, so it only works when each
                # phase contributes a scalar or a 1-D vector. A phase can
                # return a whole trajectory -- mole_conc over (time,
                # species) when a flowsheet connection converts an outlet --
                # and np.dot then fails on the shape. Contract the phase
                # axis explicitly so any trailing shape works.
                return np.tensordot(weights, values, axes=(0, 0))

            return wrapper

        # Attributes
        present = [v for v in attrs if v is not None]

        if not present:
            raise AttributeError(f"No phases have attribute {name}")

        # An attribute that only one phase carries belongs to that phase.
        # 'distrib' is the solid's in a liquid/solid slurry, and the
        # pre-refactor Slurry exposed it directly. Standing a zero in for
        # the phases that lack it is meaningless for an average and
        # outright broken for an array: np.array([0, <35 floats>]) raises
        # 'setting an array element with a sequence ... inhomogeneous
        # shape', which is how Connections.ConvertUnits crashed reading
        # len(Matter.distrib) for a crystallizer outlet. Summing or
        # averaging a single contributor returns that contributor anyway,
        # so this is only a widening of what works.
        if len(present) == 1:
            return present[0]

        values = np.array([0 if v is None else v for v in attrs])

        # Extensive properties
        if name in self.EXTENSIVE_PROPERTIES:
            return values.sum()
        if not np.issubdtype(values.dtype, np.number):
            # Some attributes are shared metadata rather than a per-phase
            # quantity -- path_data and name_species are the same object on
            # every phase of a mixture. Averaging them is meaningless, but
            # passing them through is exactly right, and old unit operations
            # read them straight off the matter they are handed.
            present = [v for v in attrs if v is not None]

            if present and all(np.array_equal(v, present[0]) for v in present):
                return present[0]

            raise AttributeError(
                f"Cannot automatically aggregate attribute '{name}'. "
                "Implement it explicitly on MixedPhase."
            )
        # Everything else defaults to mass-weighted average, contracting the
        # phase axis so per-phase arrays of any shape aggregate correctly.
        return np.tensordot(weights, values, axes=(0, 0))
    
    def getCp(self, basis='mass'):

        total_cp = 0.0

        for phase in self:

            total_cp += (
                phase.mass
                * phase.getCp(basis='mass')
            )

        return total_cp
    @property
    def name_species(self):
        return self.Phases[0].name_species
    @property
    def num_species(self):
        return len(self.name_species)
    @staticmethod
    def phase_from_legacy(legacy_phase, **mechanism_kwargs):
        """One old phase into its refactored equivalent."""
        from PharmaPy.Phases_Refactored import BasePhase

        return BasePhase.from_legacy(legacy_phase, **mechanism_kwargs)

    @staticmethod
    def legacy_slurry_volume(members):
        """
        Slurry volume of a set of old phases, for the basis conversion.

        An old SolidPhase carries an absolute distribution while this stack
        stores a number density per m3 of slurry, and the factor between them
        is this volume. It can only be worked out here, from all the phases
        together -- a solid on its own does not have one -- which is why it is
        computed at the container level and handed down to the mechanism.
        """

        total = 0.0

        for member in members:
            volume = getattr(member, 'vol', None)

            if volume:
                total += float(volume)

        return total


    @staticmethod
    def from_legacy(legacy_matter, **mechanism_kwargs):
        """An old Slurry, or a bare old phase, into this stack."""
        phases = getattr(legacy_matter, 'Phases', None)
        members = phases if isinstance(phases, (list, tuple)) \
            else [legacy_matter]

        mechanism_kwargs.setdefault(
            'vol_slurry', MixedPhase.legacy_slurry_volume(members))

        return MixedPhase([MixedPhase.phase_from_legacy(m, **mechanism_kwargs)
                           for m in members])

    def to_legacy(self):
        """An equivalent container from the pre-refactor stack.

        A liquid/solid pair becomes a Slurry, which is what the old solid
        handling units accept. A single phase converts to that phase alone,
        because the old units take a bare phase there and wrapping it in a
        Slurry would make them look for a solid that does not exist.
        """
        import PharmaPy.MixedPhases as legacy

        converted = [phase.to_legacy() for phase in self.Phases]

        if len(converted) == 1:
            return converted[0]

        families = {phase.phase_family for phase in self.Phases}

        if families != {"liquid", "solid"}:
            raise NotImplementedError(
                "The pre-refactor stack has no container for phases "
                f"{sorted(families)}; only a liquid/solid Slurry exists."
            )

        slurry = legacy.Slurry()
        slurry.Phases = converted

        return slurry

    def to_stream(self):
        mixedstream = copy.copy(self)
        phases = mixedstream._Phases
        mixedstream.__class__ = MixedStream
        mixedstream.Streams = [phase.to_stream() for phase in phases]
        return mixedstream
class MixedStream(MixedPhase):

    EXTENSIVE_PROPERTIES = {
        "mass_flow",
        "vol_flow",
        "mole_flow",
    }
    
    def __init__(self,Streams=None):
        super().__init__()
        if Streams is not None:
            self.Streams=Streams

    @property
    def Streams(self):
        # Alias for Phases but with the knowledge that they are stream objects
        return self.Phases
    @Streams.setter
    def Streams(self,value):
        self.Phases = value

    def _normalize_phases(self, streams):
        """
        Recursively flatten nested MixedStreams while preserving
        individual stream objects.
        """
        normalized = []
        for stream in streams:

            if isinstance(stream, MixedStream):
                normalized.extend(self._normalize_streams(stream.Streams))

            elif isinstance(stream,BasePhase) and stream.is_stream:
                normalized.append(stream)

            else:
                raise TypeError("All objects assigned to MixedStream must be Stream objects.")

        return normalized

    def evaluate_inputs(self, time):
        """
        Evaluate every constituent stream and return the results.
        """
        return [stream.evaluate_inputs(time) for stream in self.Streams]
    def to_legacy(self):
        """A pre-refactor SlurryStream, or the single stream alone."""
        import PharmaPy.MixedPhases as legacy

        converted = [phase.to_legacy() for phase in self.Phases]

        if len(converted) == 1:
            return converted[0]

        families = {phase.phase_family for phase in self.Phases}

        if families != {'liquid', 'solid'}:
            raise NotImplementedError(
                'The pre-refactor stack has no container for streams '
                f'{sorted(families)}; only a liquid/solid SlurryStream '
                'exists.'
            )

        slurry = legacy.SlurryStream()
        slurry.Phases = converted

        return slurry

    def to_phase(self):
        mixedphase = copy.copy(self)
        streams = mixedphase._Phases
        mixedphase.__class__ = MixedPhase
        mixedphase.Phases = [stream.to_phase() for stream in streams]
        return mixedphase  # the return was missing, so this yielded None
