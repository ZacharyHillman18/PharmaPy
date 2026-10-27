from __future__ import annotations
import copy
import string
from PharmaPy import xp as np
import os
from dataclasses import dataclass, field
from typing import Optional, Sequence, Any, ClassVar
from types import MethodType
from collections import OrderedDict
from collections.abc import Callable

import PharmaPy.Kinetics as pk
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from PharmaPy.Mechanisms import Mechanism,CrossPhaseTransferMechanism,DirectTransfer
    from PharmaPy.Phases_Refactored import BasePhase
    from PharmaPy.MixedPhases_Refactored import MixedPhase,MixedStream
## Dataclasses
@dataclass(frozen=True)
class PhaseRef:
    phase_type: str
    index: int
    def __post_init__(self):
        object.__setattr__(self, "phase_type", str(self.phase_type).lower())
    def __eq__(self,otherPhaseRef):
        return self.phase_type==otherPhaseRef.phase_type and self.index==otherPhaseRef.index

@dataclass
class PhaseConnection:
    #TODO move this to connections when done
    #active_condition checks the source sink and temp and must return a boolean
    source_phaseref: PhaseRef
    sink_phaseref: PhaseRef
    kinetics:pk.CrystKinetics|pk.RxnKinetics
    species_weights: np.ndarray | None = None
    active_condition: callable=lambda source_phase,sink_phase:True
    mechanism:"CrossPhaseTransferMechanism | None" = None



@dataclass
class PhaseMapping:

    source_phaseref: PhaseRef

    sink_phaseref: PhaseRef

@dataclass
class StreamConnection:

    stream: "MixedStream"
    phase_mappings: list[PhaseMapping]
    split_fraction: float = 1.0

class StreamConditions:

    def __init__(self, streams):
        self.streams = streams

    def __iter__(self):
        return iter(self.streams)
@dataclass
class IntraPhaseProcess:
    phaseref:PhaseRef
    mechanism: "Mechanism"

class MaterialContributionBuffer:

    __slots__ = (
        "n_values",
        "contributions",
        "rates",
        "aux",
    )

    INLET = 0
    OUTLET = 1
    INTRAPHASE = 2
    CROSSPHASE = 3

    def __init__(self, n_values):

        self.n_values = n_values

        self.contributions = np.zeros(
            (4, n_values),
            dtype=float,
        )

        self.rates = np.zeros(
            n_values,
            dtype=float,
        )

        self.aux = [[], [], [], []]

    def reset(self):
        # A traced evaluation leaves a jax array here; the next numeric one
        # needs its numpy buffer back.
        if isinstance(self.contributions, np.ndarray):
            self.contributions.fill(0.0)
        else:
            self.contributions = np.zeros((4, self.n_values), dtype=float)
        self.rates.fill(0.0)

        for aux in self.aux:
            aux.clear()

@dataclass
class StateVariable:
    name: str
    dim: int
    units: str
    state_type: str = "post"
    index: Optional[Sequence] = None
    depends_on: tuple = ("time",)
    stream: Optional[str] = None
    phaseref: Optional[PhaseRef] = None
    limit_negative_inventory: bool = True
    # If limit_negative_inventory==True, the vessel's generic material limiter checks this
    # state for negative inventory. States with their own internal
    # positivity/conservation handling may set this to False.

    compute_value: Callable[
        [
            Any,   # state_var
            float, # time
            dict,  # completed_state
            Any,   # context
        ],
        Any
    ] | None = None

    def __post_init__(self):
        if self.compute_value is None and self.state_type=='post':
            self.compute_value = self.default_compute_value

    def as_dict(self):
        """Backward compatibility."""
        out = {
            "dim": self.dim,
            "units": self.units,
            "type": self.state_type,
            "depends_on": list(self.depends_on),
        }

        if self.index is not None:
            out["index"] = self.index

        return out

    def update_variable(self, variable_name, new_value):
        setattr(self, variable_name, new_value)
    

    @staticmethod
    def default_compute_value(
        state_var,
        time,
        completed_state,
        context,
        resolved_inlets=None,
        resolved_outlets=None,
        operating_conditions=None,
    ):
        try:
            return completed_state[
                StateKey(state_var.name, state_var.phaseref)
            ]
        except KeyError:
            raise KeyError(
                f"'{state_var.name}' is not present in the completed state."
            )
    
@dataclass(frozen=True)
class StateKey:
    name: str
    phaseref: PhaseRef | None = None

@dataclass(frozen=True)
class OperatingKey:

    name: str

    connection: int | None = None

    phaseref: PhaseRef | None = None

    component: str | None = None

    port: str | None = None

    def __post_init__(self):
        object.__setattr__(self,'port',self.port.lower())



@dataclass
class ResolvedPhaseTransfer:
    connection: StreamConnection
    mapping: PhaseMapping

    vessel_phase: "BasePhase"
    stream_phase: "BasePhase"

    vol_flow: float
    species_flow: np.ndarray

    direction: str
    material_slice: slice | None = None

    # (material_slice, rate) written by this transfer's phase mechanisms.
    # The positivity limiter replays these to undo part of a contribution
    # without having to re-evaluate the mechanism.
    material_writes: list = field(default_factory=list)

    def scale(self, factor, basis):

        self.species_flow *= factor

        self.stream_phase.updatePhase(**{basis:self.species_flow})
        
@dataclass
class ResolvedStreamConnection:
    connection: StreamConnection
    transfers: list[ResolvedPhaseTransfer]

    # The stream actually carrying the resolved flows. This is the vessel's
    # working copy, not connection.stream, which stays at the template values
    # the user configured.
    stream: "MixedStream | None" = None

    def __iter__(self):
        return iter(self.transfers)

@dataclass
class TransferResult:
    state_rates: dict[StateKey]
    aux: dict
    net_mass_rate:float
@dataclass
class StateCollection:
    states: dict[StateKey, StateVariable] = field(default_factory=dict)

    # StateVariable.state_type marking a state the solver must satisfy as a
    # residual g(y) = 0 rather than advance as a derivative.
    ALGEBRAIC_STATE_TYPE: ClassVar[str] = "alg"

    # Compiled numerical layout
    _keys: tuple = field(default_factory=tuple, init=False, repr=False)
    _material_keys: tuple = field(default_factory=tuple, init=False, repr=False)
    _state_values: tuple = field(default_factory=tuple, init=False, repr=False)
    _slices: dict = field(default_factory=dict, init=False, repr=False)
    _material_slices: dict = field(default_factory=dict, init=False, repr=False)
    _algebraic_keys: tuple = field(default_factory=tuple, init=False, repr=False)
    _algebraic_mask: Any = field(default=None, init=False, repr=False)
    _dim: int = field(default=0, init=False, repr=False)
    _compiled: bool = field(default=False, init=False, repr=False)
    _material_dim: int = field(default=0, init=False, repr=False)

    def add(self, state, overwrite=False, error_on_conflict=False):
        key = StateKey(state.name, state.phaseref)

        existing = self.states.get(key)

        if existing is not None:
            if error_on_conflict:
                raise ValueError(
                    f"State {key} already exists."
                )
            if not overwrite:
                return

        self.states[key] = state
        self._compiled = False

    def names(self):
        return [key.name for key in self.states]

    def dims(self):
        return [state.dim for state in self.states.values()]

    def __contains__(self, name):
        if isinstance(name, str):
            return any(key.name == name for key in self.states)
        return name in self.states

    def compile(self):
        """
        Compile the immutable numerical layout used during integration.
        """

        keys = []
        state_values = []
        slices = {}
        material_slices = {}
        algebraic_keys = []

        start = 0
        material_start = 0
        for key, state in self.states.items():

            end = start + state.dim
            state_slice = slice(start, end)

            keys.append(key)
            state_values.append(state)
            slices[key] = state_slice

            # Material slices stay differential-only: they address the
            # inventory the positivity limiter throttles, and an algebraic
            # state has no inventory to run out of.
            if state.state_type == "diff" and key.phaseref is not None:
                material_slices[key] = slice(
                    material_start,
                    material_start + state.dim,
                )
                material_start += state.dim

            if state.state_type == self.ALGEBRAIC_STATE_TYPE:
                algebraic_keys.append(key)

            start = end

        # True where the packed vector carries a residual rather than a
        # derivative. Backends read this to build a mass matrix or an
        # implicit residual; it is all zeros for an ordinary ODE.
        algebraic_mask = np.zeros(start, dtype=bool)

        for key in algebraic_keys:
            algebraic_mask[slices[key]] = True

        self._keys = tuple(keys)
        self._material_keys = tuple(key for key in keys if key in material_slices)
        self._state_values = tuple(state_values)
        self._slices = slices
        self._material_slices = material_slices
        self._algebraic_keys = tuple(algebraic_keys)
        self._algebraic_mask = algebraic_mask
        self._dim = start
        self._material_dim = material_start
        self._compiled = True

    @property
    def algebraic_keys(self):
        """Solver states defined by a residual instead of a derivative."""
        if not self._compiled:
            self.compile()
        return self._algebraic_keys

    @property
    def algebraic_mask(self):
        if not self._compiled:
            self.compile()
        return self._algebraic_mask

    @property
    def has_algebraic(self):
        if not self._compiled:
            self.compile()
        return len(self._algebraic_keys) > 0

    @property
    def material_dim(self):
        if not self._compiled:
            self.compile()
        return self._material_dim
    @property
    def keys(self):
        if not self._compiled:
            self.compile()
        return self._keys

    @property
    def state_values(self):
        if not self._compiled:
            self.compile()
        return self._state_values

    @property
    def slices(self):
        if not self._compiled:
            self.compile()
        return self._slices

    @property
    def material_slices(self):
        if not self._compiled:
            self.compile()
        return self._material_slices

    @property
    def dim(self):
        if not self._compiled:
            self.compile()
        return self._dim
    @property
    def material_keys(self):
        if not self._compiled:
            self.compile()
        return self._material_keys
    def unpack(self, y):
        if not self._compiled:
            self.compile()

        states = {}

        for key, state_slice, state in zip(
            self._keys,
            self._slices.values(),
            self._state_values,
        ):
            value = y[state_slice]

            if state.dim == 1:
                value = value[0]

            states[key] = value

        return states

    def pack(self, state_dict):
        if not self._compiled:
            self.compile()

        y = np.empty(self._dim)

        for key, state_slice in self._slices.items():
            y[state_slice] = np.asarray(
                state_dict[key]
            ).reshape(-1)

        return y

    def unpack_history(self, y_history):
        if not self._compiled:
            self.compile()

        history = {}

        for key, state_slice, state in zip(
            self._keys,
            self._slices.values(),
            self._state_values,
        ):
            values = y_history[:, state_slice]

            if state.dim == 1:
                values = values[:, 0]

            history[key] = values

        return history

    def flatten(self, state_dict):
        flat = {}

        for key, value in state_dict.items():
            if isinstance(key, str):
                flat[key] = value
            else:
                flat[self.format_key(key)] = value

        return flat

    @staticmethod
    def format_key(key):
        if key.phaseref is None:
            return key.name
        return f"{key.name}_{key.phaseref.phase_type}{key.phaseref.index}"
@dataclass
class PhaseStateVariable:
    phaseref: PhaseRef
    state: StateVariable

@dataclass
class PhaseStateCollection:
    phasestates: dict[PhaseRef, StateCollection] = field(default_factory=dict)

    _phase_by_ref: dict = field(default_factory=dict, init=False, repr=False)
    _ref_by_phase_id: dict = field(default_factory=dict, init=False, repr=False)
    _compiled: bool = field(default=False, init=False, repr=False)

    def add(self, phase, state):
        if phase not in self.phasestates:
            self.phasestates[phase] = StateCollection()

        self.phasestates[phase].add(state)
        self._compiled = False

    def __getitem__(self, phase):
        return self.phasestates[phase]

    def __iter__(self):
        for phaseref, collection in self.phasestates.items():
            for state in collection.states.values():
                yield PhaseStateVariable(phaseref, state)

    def compile(self, phases):
        """
        Compile O(1) phase/reference lookup and each phase's state layout.
        """

        self._phase_by_ref = {}
        self._ref_by_phase_id = {}

        for phaseref, collection in self.phasestates.items():

            phase = phases.get_phase_from_ref(phaseref)

            self._phase_by_ref[phaseref] = phase
            self._ref_by_phase_id[id(phase)] = phaseref

            collection.compile()

        self._compiled = True

    def get_phase(self, phaseref):
        return self._phase_by_ref[phaseref]

    def get_ref(self, phase):
        return self._ref_by_phase_id[id(phase)]
    

@dataclass
class StateEvent:

    name: str
    function: Callable
    direction: int = 0
    terminal: bool = False
    source: Any = None
    callbakc: Any = None