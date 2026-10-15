
"""
Created on Fri July 10 2026

@author: zhillma
Refactored the code by dcasasor
"""
from PharmaPy.Phases_Refactored import BasePhase
from PharmaPy.MixedPhases_Refactored import MixedPhase, MixedStream

from PharmaPy.ProcessControl_Refactored import Controller
from PharmaPy.IntegratorBackends import ScipyBackend
from PharmaPy.Results import DynamicResult
from PharmaPy.Connections import interpolate_inputs



import copy
import numpy as np
import os
from PharmaPy.DataClasses import *
from time import perf_counter



eps = np.finfo(float).eps

class MultiPhaseVessel():

    # Declared on the class so a subclass can override it. Assigning it in
    # __init__ would shadow the subclass value with None on every instance,
    # and Connections and SimExec branch on it, so a shadowed value silently
    # takes a flowsheet down the wrong path. Spellings match the rest of
    # PharmaPy: 'Batch', 'Semibatch', 'Continuous'.
    oper_mode = None

    # Whether mechanism states carried per unit volume (a crystal population
    # per m3 of slurry) are diluted as the vessel volume changes. See
    # add_dilution_terms; the crystallizers switch it on.
    dilutes_intensive_states = False

    def __init__(self,integrator=None,temp_ref=273.15,
     isothermal=False, reset_states=False, controller=None, h_conv=0, 
      state_events={},
      adiabatic=False,Phases=None,
      basis='mass_j',ht_mode="jacket",diam=0,area_base=0,
      emit_events=True, positivity_horizon=1.0):

        # Built here rather than as a default argument: a default is
        # evaluated once at class-definition time, so every vessel in the
        # session would share one controller object, and a level controller's
        # target_volume would leak between unrelated vessels.
        if controller is None:
            controller = Controller()

        if isothermal and controller is not None:
            assert 'global_temp' not in controller.states and 'temp' not in controller.states, "Cannot change the temperature of an isothermal unit"

        self.basis = basis
        self.adiabatic = adiabatic
        self.isothermal = isothermal

        self.controller = controller #TODO ZZ refactor analyze_controls to give a Controls dataclass, an empty one if controls None

        
        # Phase init
        self._phase_connections = []
        if Phases is not None:
            self.Phases = Phases
        self._intraphase_processes = []

        #heat transfer
        self.area_ht = None
        self._Utility = None
        self.ht_mode = ht_mode
        self.h_conv = h_conv
        self.diam = diam
        self.area_base = area_base
        self.temp_ref = temp_ref #enthalpy ref

        #State events
        if state_events is None:
            state_events = []
        self.state_event_list = state_events

        # Whether mechanisms and the controller hand their switching
        # surfaces to the integrator. Root-finding pays off when a surface
        # is actually crossed and the step size is limited by that crossing;
        # it is pure overhead when the step size is limited by something
        # else, and the cost is per-step (and heavier under diffeqpy, where
        # every condition evaluation crosses into Python). Measure before
        # assuming it helps.
        self.emit_events = emit_events

        #state initialization, all types
        self._initialize_states(reset_states)
        
        #port initialization
        self._inlet_connections = []
        self._outlet_connections = []

        #Integrator
        # scipy is the only solver PharmaPy requires, so it is the default;
        # Assimulo and diffeqpy backends are opt-in. Built per vessel, like
        # the controller above, because a backend holds compiled solver state.
        if integrator is None:
            integrator = ScipyBackend()
        self.integrator = integrator

        # Window over which depletion is anticipated, in seconds.
        #
        # Both positivity guards use it: calculate_scale throttles the outlet
        # and cross-phase terms so a phase inventory survives this long, and
        # RxnKinetics scales its reaction extents on the same window. It is
        # not the solver step -- it is a fixed look-ahead, so a species whose
        # inventory would not last this long is throttled no matter how short
        # a step the integrator is actually taking.
        #
        # Shrinking it throttles less and lets a species approach zero more
        # closely; 1.0 is the value every result on this branch was produced
        # with, so it stays the default.
        self.positivity_horizon = positivity_horizon




    @property
    def Phases(self):
        return self._Phases
    
    @Phases.setter
    def Phases(self, phases):
        incoming = MixedPhase(self._as_phases(phases))

        # Connections stamps transferred_from_uo on material handed over by
        # an upstream unit. The old units merged that material into whatever
        # had already been configured -- a crystallizer is given its seed
        # solid and receives its liquid from upstream (Crystallizers.py
        # appends rather than replaces). Replacing outright would discard
        # the solid phase along with its population-balance mechanism, so
        # merge by phase family: an arriving phase supersedes the configured
        # one of the same family, and any family not present is added.
        if getattr(phases, 'transferred_from_uo', False) \
                and self._Phases is not None:

            incoming = MixedPhase(self._merge_by_family(incoming.Phases))

        self._Phases = incoming
        self._post_set_phases()

    def _as_phases(self, matter):
        """Accept held-up phases, or a stream to be converted into them.

        Connections hands a continuous unit's outlet -- a MixedStream --
        straight to the destination's Phases. A semibatch vessel accepts an
        incoming flow by definition, so converting the stream to its phase
        form is exactly right. A batch vessel does not: a continuous unit
        feeding a batch one needs a holding vessel between them to turn a
        flow into a charge, so say that rather than failing obscurely deep
        inside MixedPhase with "All phases must be PharmaPy Phase objects".
        """
        is_stream = getattr(matter, 'is_stream', False) or isinstance(
            matter, MixedStream)

        if not is_stream:
            return matter

        if self.oper_mode == 'Batch':
            raise TypeError(
                f"{type(self).__name__} is a batch unit and cannot be fed "
                "directly from a continuous one: a flow has to be "
                "accumulated into a charge first. Put a holding vessel "
                "(PharmaPy.Containers.DynamicCollector) between them."
            )

        return matter.to_phase()

    def _merge_by_family(self, arrivals):
        """Substitute arriving phases into the configured ones by family.

        Matching is positional within a family: the first arriving liquid
        replaces the first configured liquid, the second the second, and so
        on. Breaking on the first match of a family instead would let two
        arriving liquids overwrite the same slot and silently drop one.
        """
        merged = list(self._Phases.Phases)
        used = set()

        for arriving in arrivals:
            for idx, existing in enumerate(merged):
                if idx in used:
                    continue
                if existing.phase_family == arriving.phase_family:
                    merged[idx] = arriving
                    used.add(idx)
                    break
            else:
                merged.append(arriving)

        return merged
        
    

    def _basis_units(self):

        units = {
            "mass_j": "kg",
            "mass_conc": "kg/m3",
            "mole_j": "kmol",
            "mole_conc": "kmol/m3"
        }

        return units[self.basis]
    
    def _material_state_definition(self)->StateVariable:

        return StateVariable(
            name=self.basis,
            dim=self.Phases.num_species,
            units=self._basis_units(),
            index=self.Phases.name_species,
            
        )

    def _post_set_phases(self):
        self.define_material_states()
        self.initialize_defualt_states()
        self.configure_default_connections()
        self.nomenclature() 
    def configure_default_connections(self):
        """Hook for subclasses to create default outlet connections."""
        pass
    def initialize_defualt_states(self):
        self.default_diff_states_from_phases()
    def _initialize_state_collections(self):

        self.phase_states = PhaseStateCollection()

        # States exposed to solver
        self.solver_state_collection = StateCollection()

        # States exposed as outputs
        self.output_state_collection = StateCollection()

    @property
    def Inlet(self):
        raise AttributeError("Inlet is a convenience API for setting inlet_connections")


    @Inlet.setter
    def Inlet(self, inlet):

        inlet= inlet if isinstance(inlet,(list,tuple)) else [inlet]

        self._create_default_connections('inlet_connections',inlet)
    
    @property
    def inlet_connections(self)->list[StreamConnection]:
        return self._inlet_connections


    @inlet_connections.setter
    def inlet_connections(self, connections):

        if not isinstance(connections, (list,tuple)):
            raise TypeError(
                "inlet_connections should be a list or tuple"
            )

        if not all(
            isinstance(c, StreamConnection)
            for c in connections
        ):
            raise TypeError(
                "inlet_connections must contain StreamConnection objects")

        self._inlet_connections = connections
    def _create_default_connections(self,connection_attr, inlet_streams):

        connections = [
            self.make_stream_connection(stream) for stream in inlet_streams
        ]

        setattr(self,connection_attr,connections)

    @staticmethod
    def make_stream_connection(stream)->StreamConnection:
        """
        A connection carrying `stream`, each phase mapped onto the vessel
        phase of the same family and position.

        Separate from _create_default_connections so a single port can be
        replaced without rebuilding the others: a ConcurrentVessel swaps
        one inlet of a member for a flowsheet feed while the member's
        coupled inlets stay in place.
        """

        if not isinstance(stream, MixedStream):
            stream = MixedStream(stream)

        mappings = []

        counts = {}

        for phase in stream:

            phase_type = (phase.phase_family.lower())

            idx = counts.get(phase_type, 0)
            counts[phase_type] = idx + 1

            ref = PhaseRef(phase_type, idx)

            mappings.append(
                PhaseMapping(
                    source_phaseref=ref,
                    sink_phaseref=ref
                )
            )

        return StreamConnection(
            stream=stream,
            phase_mappings=mappings
        )

    @property
    def outlet_connections(self)->list[StreamConnection]:
        return self._outlet_connections
    
    @outlet_connections.setter
    def outlet_connections(self,connections):

        if not isinstance(connections, (list,tuple)):
            raise TypeError("outlet_connections should be a list or tuple")
        
        if not all(isinstance(c, StreamConnection) for c in connections):
            raise TypeError(
                "outlet_connections must contain StreamConnection objects")

        self._outlet_connections = connections

    @property
    def Outlet(self):
        """
        The unit's single outlet stream.

        After a solve this is the resolved stream, carrying the flows and
        composition the vessel actually discharged. Before one it is the
        connection's stream, which still holds the configured template.

        AttributeError raised in a property is indistinguishable from a
        missing attribute, so Python falls through to __getattr__ and the
        caller sees a bare "Outlet" instead of the reason. Every message
        here therefore says what is actually wrong.
        """

        conditions = self.outlet_conditions

        if conditions is not None and conditions.streams:

            if len(conditions.streams) == 1:
                return conditions.streams[0].stream

            raise AttributeError(
                f"This unit resolved {len(conditions.streams)} outlet "
                "streams. Use outlet_conditions to pick one."
            )

        # Backward compatibility before solve
        if len(self.outlet_connections) == 1:
            return self.outlet_connections[0].stream

        if not self.outlet_connections:
            # A batch or semibatch vessel discharges its contents rather
            # than a stream, so it legitimately has no outlet connection.
            # The old units modelled this by setting Outlet to the final
            # phase (Reactors.py: 'self.Outlet = self.Liquid_1'), which is
            # what a downstream unit in a flowsheet expects to receive.
            if self.oper_mode != 'Continuous' and self._Phases is not None:
                return self.Phases

            raise AttributeError(
                f"{type(self).__name__} has no outlet connections. "
                "Continuous units define one in "
                "configure_default_connections; batch units have none."
            )

        raise AttributeError(
            "Multiple outlet connections exist. "
            "Use outlet_connections instead.")
    
    @Outlet.setter
    def Outlet(self,outlet):
        outlet = outlet if isinstance(outlet,(list,tuple)) else [outlet]
        self._create_default_connections('outlet_connections',outlet)

    # =================================================================
    # Legacy flowsheet protocol
    #
    # SimExec and Connections were written against the pre-refactor unit
    # operations and read a handful of attributes off every unit they
    # execute. They are supplied here so a MultiPhaseVessel can sit in a
    # flowsheet next to old units. None of this is used by the vessel
    # itself; it is a translation layer and should be deleted once the
    # remaining unit operations are ported.
    # =================================================================

    # Connections.PassPhases sets this on a Semibatch destination.
    material_from_upstream = False

    # Only meaningful for units with several named outlets (none yet).
    default_output = None

    # Populated by SimExec when it walks the graph.
    bipartite = None
    names_upstream = None

    @property
    def is_continuous(self):
        """Connections uses this to decide whether the downstream unit
        receives the whole outlet trajectory or only its final point."""
        return self.oper_mode == 'Continuous'

    @property
    def states_di(self):
        """Solver state description, in the shape the old units publish.

        Results.SimulationResult skips any unit whose states_di is None, so
        with every unit refactored the summary table came out empty and
        Results.pprint called max() on it -- raising at the very end of
        SolveFlowsheet, after all the physics had already run, and leaving
        flst.result and flst.connections unassigned.

        StateVariable.as_dict already emits dim/units/type/index, which is
        exactly what that summary reads.
        """
        if self.solver_state_collection is None:
            return None

        return {
            self.solver_state_collection.format_key(key): state.as_dict()
            for key, state in self.solver_state_collection.states.items()
        }

    def flatten_states(self):
        """No-op kept for SimExec, which calls this after every solve.

        Old units accumulate a list of per-run state arrays in
        profiles_runs and stack them here. The refactored vessel already
        stores one flat array per state in self.result, so there is
        nothing to stack."""
        return self.result

    def _population_state_names(self):
        """Population states the solid phases publish, e.g. distrib or mu_n.

        Named by the mechanism rather than hardcoded, so a crystallizer
        using moments publishes its moment set the way an FVM one
        publishes its distribution. Hardcoding 'distrib' meant a moments
        crystallizer advertised a state it never emitted, and a downstream
        unit received no population at all.
        """
        names = []

        for phase in self.Phases.Phases:

            if phase.phase_family != 'solid':
                continue

            for mechanism in getattr(phase, 'mechanisms', None) or ():

                name = getattr(mechanism, 'flow_state_name', None)

                if name and name not in names:
                    names.append(name)

        return names

    @property
    def names_states_out(self):
        """State names carried by the outlet, in the vocabulary the old
        NameAnalyzer expects. Mirrors the old CSTR/BatchReactor lists."""
        names = ['mole_conc']

        names.extend(self._population_state_names())

        names.append('temp')
        names.append('vol_flow' if self.is_continuous else 'vol')

        return names

    @property
    def names_states_in(self):
        """Inlet state names, in the vocabulary the old NameAnalyzer wants.

        Not simply names_states_out. A vessel's OUTLET carries an amount
        but its INLET carries a flow, whatever the vessel's own operating
        mode. A semibatch vessel used to advertise 'vol' here, so
        NameAnalyzer had nothing to map an upstream 'vol_flow' onto and
        quietly dropped it from y_inlet -- the trajectory then arrived
        without the one state that says how fast material is entering.

        Continuous vessels already say 'vol_flow', and a batch vessel is
        never fed (Connections only builds y_inlet when the destination is
        not Batch), so in practice this only changes semibatch inlets.
        """
        names = self.names_states_out

        if self.oper_mode != 'Batch':
            names = ['vol_flow' if name == 'vol' else name
                     for name in names]

        return names

    def _outlet_fields(self):
        """
        What a downstream unit is handed, as {name: StateKey read from}.

        Names are the inter-unit vocabulary that NameAnalyzer and
        resolve_inlets speak; keys say where this vessel holds each value.
        `outputs` reads them from a finished result and `outlet_values` from
        the point last evaluated, so a recorded trajectory and a live
        coupling cannot disagree about what an outlet carries.

        A population is handed over as the solver holds it, a number DENSITY
        per m3 of slurry, which is the basis old PharmaPy moved between
        units: MSMPR.material_balances feeds the received
        u_inputs['Inlet']['distrib'] straight into
        tau_inv * (input_distrib - distrib). It used to be an absolute count
        divided by the vessel volume on the way out; that division is now
        wrong twice over.
        """
        fields = {'mole_conc': StateKey('mole_conc', PhaseRef('liquid', 0))}

        for population_name in self._population_state_names():
            fields[population_name] = StateKey(
                population_name, PhaseRef('solid', 0))

        fields['temp'] = StateKey('global_temp')

        if self.is_continuous:
            fields['vol_flow'] = StateKey('outlet_vol_flow')
        else:
            fields['vol'] = StateKey('vessel_vol')

        return fields

    @property
    def outputs(self):
        """Outlet trajectory as {state name: array}, which Connections
        attaches to the transferred matter as y_upstream.

        Returns None before a solve, which is what the old units do and
        what SimExec tests for."""
        if self.result is None:
            return None

        out = {}

        num_t = len(np.asarray(self.result.time))

        for name, key in self._outlet_fields().items():

            value = getattr(self.result, StateCollection.format_key(key), None)

            if name == 'vol_flow':
                out[name] = (
                    np.full(num_t, getattr(self.Outlet, 'vol_flow', 0.0))
                    if value is None else np.ravel(np.asarray(value)))

            elif name == 'vol':
                out[name] = (np.full(num_t, self.Phases.vol) if value is None
                             else np.ravel(np.asarray(value)))

            elif value is not None:
                out[name] = np.asarray(value, dtype=float)

        return out

    def outlet_values(self):
        """
        The outlet in the vocabulary of `outputs`, at the point the vessel
        was last evaluated, as {name: value}.

        This is the live counterpart of `outputs`: a ConcurrentVessel serves
        it to the units this vessel feeds while they are all integrated as
        one system. Every value is the one the right-hand side just used,
        including an outlet flow the positivity limiter throttled.
        """

        point = getattr(self, '_last_evaluation', None)

        if point is None:
            raise RuntimeError(
                f"{type(self).__name__}.outlet_values needs the vessel to have "
                "been evaluated: call unit_model at the point of interest "
                "first."
            )

        out = {}

        for name, key in self._outlet_fields().items():

            value = self._state_value_at(key, point)

            if value is None:
                continue

            value = np.asarray(value, dtype=float)

            out[name] = float(value) if value.ndim == 0 else value.copy()

        return out

    def _state_value_at(self, key, point):
        """
        A solver or output state at an evaluated point, or None.

        Output states are computed the way the replay computes them. Reading
        them from the completed state instead would be wrong: complete_state
        fills an output state from the phase BEFORE the phases are updated,
        so it holds the previous evaluation's value.
        """

        completed_state = point['completed_state']

        if key in self.solver_state_collection.states:
            return completed_state[key]

        variable = self.output_state_collection.states.get(key)

        if variable is not None:
            return variable.compute_value(
                state_var=variable,
                time=point['time'],
                completed_state=completed_state,
                context=self,
                resolved_inlets=point['resolved_inlets'],
                resolved_outlets=point['resolved_outlets'],
                operating_conditions=point['operating_conditions'],
            )

        return completed_state.get(key)

    

    def define_material_states(self):


        material_state = self._material_state_definition()
        counts = {}
        for i, phase in enumerate(self.Phases):
            phase_type = phase.phase_family.lower()
            idx=counts.get(phase_type,0)
            counts[phase_type] = idx+1
            phase_ref = PhaseRef(
                phase_type=phase_type,
                index=idx
            )

            self.phase_states.add(
                phase_ref,
                copy.deepcopy(material_state)
            )

    def default_diff_states_from_phases(self):
        """
        Mark the default material state for every phase as differential.

        Unit operations that require different behavior should override this
        method or modify `phase_states` after phase initialization.
        """
        # do nothing if the phases are already marked diff
        if any(
            state.state_type == "diff"
            for phasestate in self.phase_states
            for state in self.phase_states[phasestate.phaseref].states.values()
        ):
            return
        for phasestatevar in self.phase_states:
            if phasestatevar.state.name==self.basis:
                phasestatevar.state.update_variable('state_type','diff')

    @property
    def phase_connections(self)->list[PhaseConnection]:
        return self._phase_connections
    
    @phase_connections.setter
    def phase_connections(self,connections:list):
        if not all([isinstance(c,PhaseConnection) for c in connections]):
            raise TypeError("phase_connections should all be PhaseConnection objects")
        if not isinstance(connections,list):
            raise TypeError("phase_connections is expected to be a list")
        self._phase_connections = connections
        self.nomenclature(overwrite=True)

    @property
    def intraphase_processes(self)->list[IntraPhaseProcess]:
        return self._intraphase_processes
    
    @intraphase_processes.setter
    def intraphase_processes(self, regions):

        if not isinstance(regions, list):
            raise TypeError(
                "intraphase_processes is expected to be a list"
            )

        if not all(
            isinstance(r, IntraPhaseProcess)
            for r in regions
        ):
            raise TypeError(
                "regions should all be "
                "IntraphaseProcess objects"
            )

        self._intraphase_processes = regions

        self.nomenclature(overwrite=True)
    
    @property
    def Utility(self):
        return self._Utility

    @Utility.setter
    def Utility(self, utility):
        self.u_ht = 1 / (1 / self.h_conv + 1 / utility.h_conv)
        self._Utility = utility
        self.output_state_collection.add(
            StateVariable(
                name="q_ht",
                dim=1,
                units="W",
                state_type="post",
                compute_value=self.compute_qht_value
            ),overwrite=True
        )
        

    
    def complete_state(self, state:dict, time:float)->dict[StateKey]:

        completed = state.copy()

        controlled = self.controller.compute_states(
            time=time,
            completed_state=completed,
            unit=self,
        )

        completed.update(controlled)

        required_states = (
            self.solver_state_collection.states
            | self.output_state_collection.states
        )

        for statekey, variable in required_states.items():

            if statekey in completed:
                continue

            
            # Pull from phases/state sources
            if statekey == StateKey("global_temp"):
                value = self.get_default_temperature()
            else:
                value = self.get_state_value(statekey)

            if value is not None:
                completed[statekey] = value

        return completed
    def get_phase_ref(self, phase):
        for phase_ref in self.phase_states.phasestates:

            if self.Phases.get_phase_from_ref(phase_ref) is phase:
                return phase_ref

        raise ValueError(
            "The phase associated with the mechanism is not a phase "
            "belonging to this vessel."
        )
    def get_default_temperature(self):
        return self.Phases[0].temp
    
    def get_state_value(self, statekey):

        # phase-associated state
        if statekey.phaseref is not None:

            phase = self.phase_states.get_phase(statekey.phaseref)

            value = getattr(phase, statekey.name, None)

            if value is not None:
                return value

            # A phase delegates to the mechanisms in its own `mechanisms`
            # list, but a mechanism attached to an intraphase process is not
            # in that list, so its states have to be found directly.
            return self.get_mechanism_state_value(statekey)

        # unit operation state
        value = getattr(self, statekey.name, None)

        if value is not None:
            return value

        return self.get_mechanism_state_value(statekey)

    def get_mechanism_state_value(self, statekey):
        """Initial value for a solver state owned by a mechanism."""

        for mechanism in self.iter_mechanisms():

            if mechanism.owns_state(statekey):
                return getattr(mechanism, statekey.name, None)

        return None
            
    def __getattr__(self, name):
        # For Backward compatability 
        # You should not use phase_# explicitly, 
        # everywhere should always iterate over all phases
        #Exception is setting the default phase_connections 
        # since those use the same assumptions as PharmaPy 1.0
        if name.startswith("Liquid_"):
            idx = int(name.split("_")[1]) - 1
            return self.Phases.Liquids[idx]

        if name.startswith("Solid_"):
            idx = int(name.split("_")[1]) - 1
            return self.Phases.Solids[idx]

        if name.startswith("Vapor_"):
            idx = int(name.split("_")[1]) - 1
            return self.Phases.Vapors[idx]

        # A property whose getter raises AttributeError looks to Python
        # exactly like a missing attribute, so __getattr__ runs and the
        # explanation the getter raised is replaced by a bare name. Re-run
        # the getter so its own message reaches the caller.
        descriptor = getattr(type(self), name, None)

        if isinstance(descriptor, property) and descriptor.fget is not None:
            return descriptor.fget(self)

        raise AttributeError(
            f"{type(self).__name__!s} has no attribute {name!r}"
        )
       
    def _initialize_states(self,reset=False):

        self.reset_states = reset

        # Populated lazily by unit_model, but update_phases_from_state can run
        # before the first right-hand side evaluation (reset, replay).
        self._timers = {}

        self.state_variables = StateCollection()

        self.input_states = StateCollection()

        self.output_states = StateCollection()

        self.elapsed_time = 0

        self.result = None
        self._initial_solver_state = None

        # Resolved outlet streams, available once the unit has been solved.
        self.outlet_conditions = None

        # Memo for evaluate_events, keyed on the exact (time, states).
        self._event_cache_key = None
        self._event_cache_values = None

        # The point unit_model last evaluated (time, completed state, resolved
        # streams), which outlet_values reads. None until the first call.
        self._last_evaluation = None
        self._last_resolved_outlets = None

        self._initialize_state_collections()

        
        
    def reset(self):

        self.controller.reset()

        self.elapsed_time = 0
        self.result = None
        self._last_evaluation = None

        if self._initial_solver_state is not None:
            self.update_phases_from_state(self._initial_solver_state)

        for process in self.intraphase_processes:
            process.mechanism.reset()

        for connection in self.phase_connections:
            if connection.mechanism is not None:
                connection.mechanism.reset()
    
    @property
    def has_energy_balance(self):
        anytemp = any(StateKey('temp',phaseref) in self.controller.states for phaseref in self.phase_states.phasestates.keys())
        return (
            not self.isothermal
            and StateKey("global_temp") not in self.controller.states and not anytemp
        )
    @property
    def has_algebraic_balance(self):
        """
        True when some mechanism contributes a residual rather than a rate.

        Checked the way has_energy_balance is: when it is False the whole
        algebraic pass is skipped, so an ordinary ODE never pays for the
        machinery or allocates a vector of zeros to carry nothing.
        """
        return self.solver_state_collection.has_algebraic

    @property
    def has_utility_balance(self):
        return (
            self.has_energy_balance
            and self.Utility is not None
            and not self.adiabatic
        )
    
    def define_solver_states(self,overwrite=False):

        
        for phase_state in self.phase_states:
            if phase_state.state.state_type!='diff': 
                continue
            state_copy = copy.deepcopy(phase_state.state)
            state_copy.phaseref = (phase_state.phaseref)
            self.solver_state_collection.add(state_copy,overwrite)
        
        if self.has_energy_balance:

            self.solver_state_collection.add(
                StateVariable(
                    name="global_temp",
                    dim=1,
                    units="K",
                    state_type='diff'
                ),overwrite
            )
            
        for connection in self.phase_connections:

            if connection.mechanism is not None:

                mechanism = connection.mechanism

                phase_ref = None

                if mechanism.owning_phase is not None:
                    phase_ref = self.get_phase_ref(mechanism.owning_phase)

                mechanism.add_solver_state_variables(
                    self.solver_state_collection,
                    overwrite=overwrite,
                    phase_ref=phase_ref,
                )

        for process in self.intraphase_processes:
            # An intraphase mechanism belongs to the phase its process runs
            # in, so that is the owner of any solver state it declares.
            process.mechanism.add_solver_state_variables(
                self.solver_state_collection,
                overwrite=overwrite,
                phase_ref=process.phaseref,
            )
    def define_output_states(self,overwrite=False):
        self.recorded_output_history ={}
        self.output_state_collection.add(
            StateVariable(
                name="Total_m_in_vessel",
                dim=1,
                units="kg/s",
                state_type="post",
                compute_value=self.compute_total_mass_in_vessel
            ),overwrite
        )

        # Reported so that a downstream unit reading this unit's outputs
        # gets the real trajectory. These used to be fabricated as flat
        # arrays from the final value, which made a level-controlled outlet
        # look like a constant flow.
        self.output_state_collection.add(
            StateVariable(
                name="vessel_vol",
                dim=1,
                units="m3",
                state_type="post",
                compute_value=self.compute_vessel_volume,
            ),overwrite
        )

        if self.outlet_connections:
            self.output_state_collection.add(
                StateVariable(
                    name="outlet_vol_flow",
                    dim=1,
                    units="m3/s",
                    state_type="post",
                    compute_value=self.compute_outlet_volflow_value,
                ),overwrite
            )

        if not self.has_energy_balance:
            self.output_state_collection.add(
                StateVariable(
                                name="global_temp",
                                dim=1,
                                units="K",
                                state_type='post',
                            ),overwrite
                        )
        
        # Every liquid phase reports its composition, whether or not it
        # happens to carry a reaction. This used to sit inside the
        # intraphase_processes loop below, so a crystallizer -- which has
        # no reaction -- registered no mole_conc output at all, even though
        # names_states_out advertises one. A downstream unit was then handed
        # an outlet trajectory with no liquid composition in it.
        counts = {}

        for phase in self.Phases.Phases:

            phase_type = phase.phase_family.lower()
            idx = counts.get(phase_type, 0)
            counts[phase_type] = idx + 1

            if phase_type != 'liquid':
                continue

            self.output_state_collection.add(
                StateVariable(
                    name="mole_conc",
                    dim=phase.num_species,
                    units="kmol/m3",
                    state_type="post",
                    index=phase.name_species,
                    phaseref=PhaseRef(phase_type, idx),
                    compute_value=self.compute_mole_conc_value
                ),
                overwrite
            )

        for process in self.intraphase_processes:

            process.mechanism.add_output_state_variables(
                self.output_state_collection,
                overwrite=overwrite,
                process=process
            )

        

        for conn in self.phase_connections:

            if conn.mechanism is not None:
                conn.mechanism.add_output_state_variables(
                    self.output_state_collection,overwrite
                )

    def nomenclature(self,overwrite=False):

        self.define_solver_states(overwrite)
        self.define_output_states(overwrite)
        self.name_states = self.solver_state_collection.names()
        self.dim_states = self.solver_state_collection.dims()


    @staticmethod
    def compute_total_mass_in_vessel(
            state_var,
            time,
            completed_state,
            context,
            resolved_inlets=None,
            resolved_outlets=None,
            operating_conditions=None,
        ):
        total_mass = 0
        for phase in context.Phases:
            total_mass+= phase.mass
        return total_mass
        
    @staticmethod
    def compute_vessel_volume(
            state_var,
            time,
            completed_state,
            context,
            resolved_inlets=None,
            resolved_outlets=None,
            operating_conditions=None,
        ):
        """Holdup volume at this time point.

        The replay updates the phases from the solver state before the
        output sweep runs, so reading phase.vol here gives the value at this
        time rather than at the end of the run.
        """
        return sum(phase.vol for phase in context.Phases)

    @staticmethod
    def compute_outlet_volflow_value(
            state_var,
            time,
            completed_state,
            context,
            resolved_inlets=None,
            resolved_outlets=None,
            operating_conditions=None,
        ):
        """Volumetric flow actually discharged at this time point.

        Taken from the resolved transfers rather than from the controller's
        request, so it reflects any rescaling the positivity limiter applied.
        """
        if resolved_outlets is None:
            return 0.0

        v_flow = 0.0

        for connection in resolved_outlets.streams:
            for transfer in connection:
                v_flow += transfer.vol_flow

        return v_flow
    @staticmethod
    def compute_outlet_massflow_value(
            state_var,
            time,
            completed_state,
            context,
            resolved_inlets=None,
            resolved_outlets=None,
            operating_conditions=None,
        ):

        if resolved_outlets is None:
            return 0.0

        m_flow = 0.0

        for connection in resolved_outlets.streams:
            for transfer in connection:
                m_flow += transfer.species_flow.sum()

        return m_flow
    
        
    @staticmethod
    def compute_qht_value(
            state_var,
            time,
            completed_state,
            context,
            resolved_inlets=None,
            resolved_outlets=None,
            operating_conditions=None,
        ):
        temp = completed_state[StateKey("global_temp")]
        temp_ht = context.Utility.temp_in

        return context.get_heat_transfer_rate(
            temp,
            temp_ht,
        )
    
    @staticmethod
    def compute_mole_conc_value(
            state_var,
            time,
            completed_state,
            context,
            resolved_inlets=None,
            resolved_outlets=None,
            operating_conditions=None,
        ):

        phase_ref = state_var.phaseref

        mass_key = StateKey(
            context.basis,
            phase_ref,
        )

        mass = completed_state[mass_key]

        phase = context.phase_states.get_phase(
            phase_ref
        )

        # kmol/m3
        return mass / phase.mw / phase.vol
    def update_phases_from_state(self, completed_state):

        global_temp = completed_state.get(StateKey("global_temp"))

        for phase_ref,collection in self.phase_states.phasestates.items():

            phase = self.phase_states.get_phase(phase_ref)

            updates = {}

            for variable in collection.states.values():
                key = StateKey(variable.name, phase_ref)

                if key in completed_state:
                    updates[variable.name] = completed_state[key]

            if global_temp is not None and "temp" not in updates:
                updates["temp"] = global_temp
            t0 = perf_counter()
            phase.update_from_solver_state(
                updates,
                completed_state,
                unit=self
            )
            self._timers['update_phases_from_solver_state'] = self._timers.get('update_phases_from_solver_state',0)+perf_counter()-t0

        for mechanism in getattr(self, "_workspace_mechanisms", ()):
            mechanism.update_state(completed_state, unit=self)

    def pack_state_rates(self, material_rates=None, global_rates=None,
                         algebraic_residuals=None):
        """
        Pack one vector the solver can consume.

        Differential slots carry the derivative; algebraic slots carry the
        residual g(y), which must be zero at a consistent solution. Keeping
        both in one vector of the same length means the backends differ only
        in how they interpret it (mass matrix, or an implicit residual), and
        unit_model's return shape never changes.

        Every solver state is visited, not just the phase-owned ones.
        Iterating the material keys alone leaves a state with no owning
        phase, currently the vessel temperature, at the zero the buffer was
        filled with, silently discarding the energy balance. material_rates
        is optional because unit_model requests the energy balance on its
        own, in which case only global_rates is supplied and the material
        slots stay zero.
        """

        buffer = self._solver_rate_buffer
        buffer.fill(0.0)

        material_slices = self.solver_state_collection.material_slices
        solver_slices = self.solver_state_collection.slices

        for key in self.solver_state_collection.keys:

            if algebraic_residuals is not None and key in algebraic_residuals:
                buffer[solver_slices[key]] = np.asarray(
                    algebraic_residuals[key]
                ).reshape(-1)
                continue

            if material_rates is not None and key in material_slices:
                buffer[solver_slices[key]] = material_rates[material_slices[key]]
                continue

            if global_rates is not None and key in global_rates:
                buffer[solver_slices[key]] = np.asarray(
                    global_rates[key]
                ).reshape(-1)
                continue

            # unit_model's diagnostic paths (mat_bce, enrgy_bce) deliberately
            # supply one half of the balances only, and the slots the other
            # half would have filled keep the zero the buffer was filled with.
            # When both halves were supplied, a key neither produced is a real
            # packing error and still raises.
            if material_rates is not None and global_rates is not None:
                raise KeyError(
                    f"StateKey {key} not found in material_rates or "
                    "global_rates"
                )

        return buffer

    def save_initial_solver_state(self, states=None, time=None):
        """
        Save the completed solver state corresponding to the initial condition.

        The state is saved only if an initial state has not already been
        established. This prevents subsequent solves from redefining the
        original state after the phases have mutated.
        """
        if hasattr(self, "_initial_solver_state") and self._initial_solver_state is not None:
            return self._initial_solver_state

        if states is None:
            states = self.create_solver_init_states()

        if time is None:
            time = self.elapsed_time

        unpacked_state = self.solver_state_collection.unpack(states)
        completed_state = self.complete_state(unpacked_state,time)

        self._initial_solver_state = completed_state

        return completed_state
    def unit_model(self, time, states, params=None, sw=None,
                    mat_bce=False, enrgy_bce=False,alg_bce=False, limiter_dt=None):
        if not hasattr(self, "model_call_count"):
            self.model_call_count = 0
            self._timers={}
        self.model_call_count += 1
        limiter_dt = (
            limiter_dt if limiter_dt is not None else self.positivity_horizon
        )
        t0=perf_counter()
        unpacked_state = self.solver_state_collection.unpack(states)
        self._timers['unpack'] = self._timers.get('unpack',0)+perf_counter()-t0

        t0=perf_counter()
        completed_state = self.complete_state(unpacked_state,time)
        self._timers['complete_state'] = self._timers.get('complete_state',0)+perf_counter()-t0

        t0=perf_counter()
        self.update_phases_from_state(completed_state)
        self._timers['update_phases'] = self._timers.get('update_phases',0)+perf_counter()-t0
        # Balances
        t0=perf_counter()
        material_rates, material_buffer = self.material_balances(
            time,completed_state, limiter_dt=limiter_dt)
        self._timers['material_balances_total'] = self._timers.get('material_balances_total',0)+perf_counter()-t0

        if mat_bce:
            return self.pack_state_rates(material_rates)
        global_rates = {}
        if self.has_energy_balance:
            t0 = perf_counter()
            energy_rates = self.energy_balances(time,completed_state, material_buffer)
            self._timers['energy_balances_total'] = self._timers.get('energy_balances_total',0)+perf_counter()-t0
            global_rates.update(energy_rates)

        # utility_rates = self.utility_energy_balance(
        #     time,completed_state)
        # global_rates.update(utility_rates)

        if enrgy_bce:
            return self.pack_state_rates(global_rates=global_rates)

        # Skipped entirely, like the energy balance, when no mechanism
        # declares an algebraic state: an ordinary ODE never builds or
        # carries a residual vector.
        algebraic_residuals = None

        if self.has_algebraic_balance:
            t0 = perf_counter()
            algebraic_residuals = self.algebraic_balances(time, completed_state)
            self._timers['algebraic_balances_total'] = self._timers.get('algebraic_balances_total',0)+perf_counter()-t0

            if alg_bce:
                return algebraic_residuals

        t0 = perf_counter()
        balances = self.pack_state_rates(material_rates=material_rates,
                                        global_rates=global_rates,
                                        algebraic_residuals=algebraic_residuals)
        self._timers['pack_state_rates'] = self._timers.get('pack_state_rates',0)+perf_counter()-t0
        assert len(balances) == len(states), (
            f"Returned {len(balances)} derivatives "
            f"for {len(states)} solver states."
        )

        self.derivatives = balances
        return balances

    def compile_structure(self):

        """Compiles the array-based representation of the unit operation's state variables and their relationships to the underlying phases and mechanisms.
        This method should be called after the unit operation's phases, mechanisms, and state variables have been defined, but before any simulation is run.
        It prepares the internal data structures for efficient numerical computation."""

        # State layouts
        self.solver_state_collection.compile()
        self.output_state_collection.compile()

        # Phase layouts
        self.phase_states.compile(self.Phases)

        # Mechanisms that run their own positivity guard read the horizon off
        # themselves rather than taking it as an argument, which would mean a
        # new keyword on every get_solver_state_rates signature. Refreshed
        # here rather than at construction because compile_structure runs at
        # the top of every solve, so a horizon changed after the mechanism was
        # built still reaches it.
        for mechanism in self.iter_mechanisms():
            mechanism.positivity_horizon = self.positivity_horizon

        # Persistent numerical buffers
        self._material_contributions = MaterialContributionBuffer(
            self.solver_state_collection.material_dim
        )

        self._solver_rate_buffer = np.empty(
            self.solver_state_collection.dim
        )

        # PhaseRef -> this phase's material inventory slice
        self._material_slice_by_phase = {}

        for key, material_slice in (
            self.solver_state_collection.material_slices.items()
        ):
            if key.name == self.basis:
                self._material_slice_by_phase[key.phaseref] = material_slice

        # Default cross-phase transfer mechanisms
        for connection in self._phase_connections:
            if connection.mechanism is None:
                connection.mechanism = DirectTransfer()

        self._compile_stream_workspaces()
        self._compile_positivity_layout()
        self.compile_events()

    def _compile_stream_workspaces(self):
        """
        Allocate one reusable stream per connection.

        Resolving a connection used to deep-copy its stream on every
        right-hand side evaluation. For a stream carrying a discretized
        phase that copy dominated the run time, so each connection now owns
        a single scratch stream that is overwritten in place instead. The
        copy still exists, so the user's connection stream is never mutated;
        it is just made once rather than tens of thousands of times.
        """

        self._inlet_workspaces = [
            copy.deepcopy(connection.stream)
            for connection in self.inlet_connections
        ]

        self._outlet_workspaces = [
            copy.deepcopy(connection.stream)
            for connection in self.outlet_connections
        ]

        # Which inlet fields a controller last overrode, per connection and
        # phase, so they can be put back when it stops overriding them.
        self._inlet_overrides_applied = {}

        # Upstream trajectory evaluated at one time, reused across the
        # several inlet resolutions that share it. Cleared here so a new
        # solve cannot read a value cached at the same time from the last.
        self._inlet_traj_cache = None

        # A stream phase shares its vessel phase's mechanisms by reference
        # (to_stream is a shallow copy), so the workspaces hold their own
        # copies and those copies have to be advanced alongside the vessel's.
        # Sharing the live mechanism instead would let resolving an outlet
        # write back into the vessel's own distribution.
        self._workspace_mechanisms = [
            mechanism
            for workspaces in (self._inlet_workspaces, self._outlet_workspaces)
            for workspace in workspaces
            for phase in workspace
            for mechanism in phase.mechanisms
        ]

    def _evaluate_inlet_trajectories(self, time):
        """Upstream inlet values at this time, or None if every feed is static.

        Connections faithfully records an upstream unit's outlet trajectory
        on the stream it hands over, but nothing here ever read it, so a
        downstream continuous unit ran on the upstream's *final* composition
        from t=0 onwards -- dimensionally sane, quietly wrong.

        Two traps. The trajectory lives on the inner phase: Inlet wraps the
        stream in a MixedStream, which declares y_upstream/y_inlet/
        time_upstream as class attributes, so reading them off the wrapper
        always yields None. And this runs twice per right-hand side
        evaluation, so the result is cached per time -- interpolate_inputs
        builds a fresh interpolant per key per call.
        """
        cached = self._inlet_traj_cache

        if cached is not None and cached[0] == time:
            return cached[1]

        out = None

        for connection_num, connection in enumerate(self.inlet_connections):

            for mapping in connection.phase_mappings:

                # Connections writes the trajectory onto whatever object it
                # was handed. When the upstream outlet is a MixedStream that
                # is the wrapper; when it is a bare stream it is the phase
                # itself. Check both rather than assuming which.
                inner = connection.stream.get_phase_from_ref(
                    mapping.source_phaseref)

                values = self._inlet_values_at(inner, time)

                if values is None:
                    values = self._inlet_values_at(connection.stream, time)

                if values is None:
                    continue

                out = {} if out is None else out
                out[(connection_num, mapping.source_phaseref)] = values

        self._inlet_traj_cache = (time, out)

        return out

    def invalidate_inlet_cache(self):
        """
        Forget the inlet values cached for the current time.

        The cache is keyed on time alone, which is right for a recorded
        trajectory or a time profile: those depend on nothing else. An inlet
        coupled live to another vessel also depends on that vessel's state,
        which changes at a fixed time whenever the integrator perturbs it, so
        a ConcurrentVessel calls this before every evaluation of the member.
        """
        self._inlet_traj_cache = None

    def _inlet_values_at(self, source, time):
        """Inlet field values at one time, or None when the feed is static."""
        dynamic = getattr(source, "DynamicInlet", None)

        if dynamic is not None:
            return dict(source.evaluate_inputs(time))

        if getattr(source, "y_upstream", None) is None:
            return None

        y_inlet = getattr(source, "y_inlet", None)

        if not y_inlet:
            return None

        # A crystal size distribution used to be refused here, because it
        # belongs to a mechanism rather than to the phase and so cannot go
        # through updatePhase. _apply_mechanism_inlet_states now routes it
        # to the owning mechanism instead, so it is interpolated with
        # everything else.
        times = getattr(source, "time_upstream", None)

        # A batch source reports a single time rather than a trajectory
        # (Connections.FeedConnection), so there is nothing to interpolate.
        if np.ndim(times) == 0:
            return dict(y_inlet)

        return {
            name: interpolate_inputs(time, times, values)
            for name, values in y_inlet.items()
        }

    def _apply_inlet_overrides(
        self,
        stream_phase,
        template_phase,
        ops,
        workspace_key,
        baseline=None,
    ):
        """
        Apply the upstream trajectory and controller overrides to a reused
        inlet stream.

        The workspace persists between calls, so a field the controller
        overrode on one evaluation and left alone on the next has to be
        restored. Restoring it from the connection's static template would
        overwrite the trajectory, so the trajectory is the baseline where one
        exists and the template supplies the rest. Controller overrides are
        applied last and still win.
        """
        baseline = baseline or {}

        # States owned by a mechanism cannot go through updatePhase, so
        # peel them off first and let the mechanism take them.
        baseline = self._apply_mechanism_inlet_states(stream_phase, baseline)
        ops = self._apply_mechanism_inlet_states(stream_phase, ops)

        previous = self._inlet_overrides_applied.get(workspace_key)

        if previous:

            stale = previous - ops.keys()

            if stale:
                stream_phase.updatePhase(
                    **{
                        name: (baseline[name] if name in baseline
                               else getattr(template_phase, name))
                        for name in stale
                    }
                )

        # One call, not one per field: updatePhase sets composition before
        # amount, and volume is converted to mass using the density implied
        # by that composition. Split across calls it would use a stale one.
        arriving = {name: value for name, value in baseline.items()
                    if name not in ops}

        if arriving:
            stream_phase.updatePhase(**arriving)

        if ops:
            stream_phase.updatePhase(**ops)

        self._inlet_overrides_applied[workspace_key] = set(ops)

    def _mechanism_state_names(self):
        """Names of solver states owned by a mechanism rather than a phase."""
        names = set()

        for phase in self.Phases.Phases:

            for mechanism in getattr(phase, 'mechanisms', None) or ():

                for attr in ('distribution_state_name',
                             'moments_state_name'):

                    name = getattr(mechanism, attr, None)

                    if name:
                        names.add(name)

        return names

    def _apply_mechanism_inlet_states(self, stream_phase, values):
        """Hand mechanism-owned inlet states to the mechanism.

        Returns the remaining states for the normal updatePhase path. Two
        things happen here: a state this phase's mechanism owns (a crystal
        size distribution) is applied to that mechanism, and a
        mechanism-owned state belonging to a DIFFERENT phase is dropped.
        The second matters because a trajectory is recorded per connection
        rather than per phase, so the liquid mapping is handed the solid's
        distribution as well, and updatePhase would reject it.
        """
        if not values:
            return values

        owned = {}

        for mechanism in getattr(stream_phase, 'mechanisms', None) or ():

            for attr in ('distribution_state_name', 'moments_state_name'):

                name = getattr(mechanism, attr, None)

                if name and name in values:
                    owned[name] = mechanism

        foreign = {name for name in values
                   if name in self._mechanism_state_names()
                   and name not in owned}

        # Same reason, the other way round: the solid mapping is handed
        # the liquid's concentration. A solid has no solvent, so its
        # mole_conc setter raises asking which species that is -- the
        # state simply is not the solid's to take.
        if getattr(stream_phase, 'phase_family', '').lower() != 'liquid':
            # vol_flow is a connection-level quantity describing the whole
            # stream -- resolve_inlets already skips it per phase for the
            # controller ops, for the same reason. Applied to a solid it
            # sets a volume the distribution does not support, and the
            # mechanism's set_mass override raises 'cannot scale a
            # distribution with zero third moment'. The solid's content
            # comes from its distribution, not from a volume.
            foreign |= {name for name in values
                        if name in ('mole_conc', 'mass_conc', 'vol_flow',
                                    'vol', 'mass_flow', 'mass')}

        if not owned and not foreign:
            return values

        for name, mechanism in owned.items():
            mechanism.apply_inlet_state(name, values[name])

        return {name: value for name, value in values.items()
                if name not in owned and name not in foreign}

    def _inlet_phase_shares(self, connection, trajectory):
        """Volumetric flow carried by each phase of a feed.

        An upstream unit publishes vol_flow for the WHOLE stream, so
        handing that value to every phase counts it once per phase. For a
        slurry feed that delivered liquid + solid = 1.55x the upstream
        outlet flow, and overstated the crystal feed by the same factor.
        Splitting it restores the invariant the rest of resolve_inlets
        assumes: each phase vol_flow is that phase share, and their sum is
        the stream flow.

        The solid share is its volume fraction, which the distribution
        gives directly: it is published per m3 of slurry, so
        kv * mu_3 * VOLUME_UNIT_FACTOR is already a fraction.

        Returns None when there is nothing to split -- a static feed, or a
        single-phase one whose share is the whole flow either way.
        """
        if not trajectory or 'vol_flow' not in trajectory:
            return None

        refs = [m.source_phaseref for m in connection.phase_mappings]

        if len(refs) < 2:
            return None

        total_flow = float(trajectory['vol_flow'])
        shares = {}
        solid_fraction = 0.0

        for ref in refs:

            if ref.phase_type == 'liquid':
                continue

            fraction = 0.0

            try:
                phase = self.phase_states.get_phase(ref)
            except KeyError:
                phase = None

            for mechanism in getattr(phase, 'mechanisms', None) or ():

                name = getattr(mechanism, 'flow_state_name', None)

                if name is None or name not in trajectory:
                    continue

                # The mechanism knows how to read its own population: an
                # integral over a resolved distribution, or mu_3 directly.
                fraction = mechanism.solid_volume_fraction(
                    np.asarray(trajectory[name], dtype=float))

            shares[ref] = max(fraction, 0.0)
            solid_fraction += shares[ref]

        solid_fraction = min(solid_fraction, 1.0)

        remaining = [ref for ref in refs if ref not in shares]

        if remaining:
            each = (1.0 - solid_fraction) / len(remaining)

            for ref in remaining:
                shares[ref] = each

        return {ref: share * total_flow for ref, share in shares.items()}

    def _get_inlet_workspace(self, connection_num, connection):

        workspaces = getattr(self, "_inlet_workspaces", None)

        if workspaces is None or connection_num >= len(workspaces):
            self._compile_stream_workspaces()
            workspaces = self._inlet_workspaces

        return workspaces[connection_num]

    def _get_outlet_workspace(self, connection_num, connection):

        workspaces = getattr(self, "_outlet_workspaces", None)

        if workspaces is None or connection_num >= len(workspaces):
            self._compile_stream_workspaces()
            workspaces = self._outlet_workspaces

        return workspaces[connection_num]

    def _compile_positivity_layout(self):
        """
        Precompute the flat layout used by the positivity limiter.

        The limiter runs on every right-hand side evaluation, so everything
        that depends on the state layout rather than on the state values is
        resolved once, here.
        """

        collection = self.solver_state_collection

        limited = np.zeros(collection.material_dim, dtype=bool)
        groups = []

        for key, material_slice in collection.material_slices.items():

            if not collection.states[key].limit_negative_inventory:
                continue

            limited[material_slice] = True
            groups.append((key, material_slice))

        self._positivity_limited = limited
        self._positivity_groups = tuple(groups)
        self._has_positivity_limits = bool(groups)

        self._inventory_buffer = np.zeros(collection.material_dim)
        self._scale_buffer = np.ones(collection.material_dim)

    def compile_integrator(self, **kwargs):
        self.compile_structure()
        return self.integrator.compile_integrator(
            self,
            **kwargs
        ) # TODO I don't think this gets called, though it may if the user wants to use fast solve and not solve_unit

    # Neutral linear-solver request ("krylov", "dense", "sparse"), or None to
    # keep the backend's default. Declared rather than applied by calling
    # self.integrator, so that something solving several vessels as one
    # system (ConcurrentVessel) can collect every member's request and pass
    # the right one to the integrator actually doing the work.
    linear_solver_hint = None

    def configure_solver(self):
        if self.linear_solver_hint is not None:
            self.integrator.set_linear_solver(self.linear_solver_hint)

    def create_solver_init_states(self):
        return self.solver_state_collection.pack(
            self.complete_state({},0))

    def solve_unit(
        self,
        runtime=None,
        time_grid=None,
        **kwargs,
    ):
        self.compile_structure()
        return self.integrator.solve(
            self,
            runtime=runtime,
            time_grid=time_grid,
            **kwargs,
        )

    
    def get_operating_conditions(self,time:float,completed_state:dict[StateKey])->tuple[StreamConditions,StreamConditions,dict[OperatingKey]]:
        # --------------------------------------------
        # Controller sees current vessel state
        # --------------------------------------------

        self.controller.observe(time,completed_state,self)

        operating_conditions=self.controller.compute_operating_conditions(
                time,
                completed_state,
                self
            )
        # --------------------------------------------
        # Resolve inlet after possible inlet control
        # --------------------------------------------
        inlet_inputs = self._evaluate_inlet_trajectories(time)

        resolved_inlets = self.resolve_inlets(completed_state,operating_conditions,inlet_inputs)
        # --------------------------------------------
        # Controller observes actual inlet
        # --------------------------------------------
        self.controller.observe(time,completed_state,self,resolved_inlets=resolved_inlets)


        operating_conditions.update(
            self.controller.compute_operating_conditions(
                time,completed_state,self,resolved_inlets=resolved_inlets))


        # Re-resolve inlet in case controller changed it
        resolved_inlets = self.resolve_inlets(completed_state,operating_conditions,inlet_inputs)

        return resolved_inlets,operating_conditions

    def resolve_current_inlets(self, time, completed_state)->StreamConditions:
        """
        The inlets exactly as the right-hand side resolves them at this point.

        For code that runs outside the resolve cycle, chiefly event
        functions, which are handed no resolved streams. Reading the inlet
        connections directly gives the configured template instead, which
        is wrong whenever the feed follows an upstream trajectory, a live
        coupling or a controller override.
        """
        resolved_inlets, _ = self.get_operating_conditions(
            time, completed_state)

        return resolved_inlets

    def iter_mechanisms(self):
        """
        Every mechanism attached to this vessel, each yielded once.

        A phase mechanism is commonly also reachable through a phase
        connection wrapped in a MechanismView, so identity is tracked through
        the wrapper to avoid registering the same events or residuals twice.
        """

        seen = set()

        for phase in self.Phases:

            for mechanism in getattr(phase, "mechanisms", ()) or ():

                if id(mechanism) not in seen:
                    seen.add(id(mechanism))
                    yield mechanism

        for connection in self.phase_connections:

            mechanism = connection.mechanism

            if mechanism is None:
                continue

            # MechanismView delegates attribute access, so unwrap before
            # testing identity against the phase's own mechanism list.
            underlying = getattr(mechanism, "mechanism", mechanism)

            if id(underlying) not in seen:
                seen.add(id(underlying))
                yield mechanism

        for process in self.intraphase_processes:

            if id(process.mechanism) not in seen:
                seen.add(id(process.mechanism))
                yield process.mechanism

    def get_events(self):
        """
        Switching surfaces declared by the controller and the mechanisms.

        Handing these to the integrator lets it locate a regime change by
        root-finding and restart cleanly there, instead of discovering it by
        failing steps. That is what makes a small maxh unnecessary.
        """

        if not self.emit_events:
            return []

        events = list(self.controller.get_events(self))

        for mechanism in self.iter_mechanisms():
            events.extend(mechanism.get_events(self))

        # Events supplied directly to the constructor, when they use the
        # StateEvent contract rather than the legacy dictionary format that
        # Commons.eval_state_events consumes.
        for event in self.state_event_list or ():
            if isinstance(event, StateEvent):
                events.append(event)

        return events

    def compile_events(self):
        self._compiled_events = tuple(self.get_events())
        self._event_cache_key = None
        self._event_cache_values = None
        return self._compiled_events

    @property
    def compiled_events(self):
        events = getattr(self, "_compiled_events", None)

        if events is None:
            events = self.compile_events()

        return events

    @property
    def has_events(self):
        return len(self.compiled_events) > 0

    def evaluate_events(self, time, states):
        """
        Evaluate every event function at (time, states).

        The phases are brought up to date first, because event functions are
        written against the vessel (unit.Phases.vol, a mechanism's
        supersaturation) rather than against the raw solver vector. This runs
        once per solver step, not once per right-hand side evaluation.
        """

        events = self.compiled_events

        if not events:
            return np.zeros(0)

        states = np.asarray(states, dtype=float)

        # Every event is evaluated together and the result memoized against
        # the exact (time, state) it came from. SciML asks one callback at a
        # time, so without this a vessel with N events would rebuild the
        # phases N times per step to answer N questions about the same point.
        # Hashing the raw bytes is exact, and cheap next to complete_state.
        cache_key = (time, states.tobytes())

        if self._event_cache_key == cache_key:
            return self._event_cache_values

        unpacked = self.solver_state_collection.unpack(states)
        completed_state = self.complete_state(unpacked, time)
        self.update_phases_from_state(completed_state)

        values = np.empty(len(events))

        for index, event in enumerate(events):
            values[index] = event.function(time, completed_state, self)

        self._event_cache_key = cache_key
        self._event_cache_values = values

        return values

    def handle_event(self, time, event_indices):
        """
        Respond to located events.

        Non-terminal events need no action: the integrator has already
        stopped at the root and will restart there, which is the whole point.
        Returns True when the run should stop.
        """

        events = self.compiled_events

        return any(
            events[index].terminal
            for index in event_indices
            if 0 <= index < len(events)
        )

    
    def algebraic_balances(self, time, completed_state):
        """
        Residuals for solver states defined by a constraint, not a rate.

        Each mechanism returns {StateKey: residual}; a consistent solution is
        one where every residual is zero. Only called when
        has_algebraic_balance is True.
        """

        residuals = {}

        for mechanism in self.iter_mechanisms():

            contribution = mechanism.get_solver_state_residuals(
                time=time,
                completed_state=completed_state,
                unit=self,
            )

            if contribution:
                residuals.update(contribution)

        missing = [
            key for key in self.solver_state_collection.algebraic_keys
            if key not in residuals
        ]

        if missing:
            raise KeyError(
                "No mechanism supplied a residual for algebraic state(s) "
                f"{[str(key) for key in missing]}. Every state registered "
                "with state_type='alg' must be returned by some mechanism's "
                "get_solver_state_residuals."
            )

        return residuals

    def material_balances(
        self,
        time:float,
        completed_state:dict[StateKey],
        limiter_dt=1.0
    ):

        resolved_inlets,operating_conditions = self.get_operating_conditions(time,completed_state)
        # --------------------------------------------
        # Material balances
        # --------------------------------------------
        buffer = self.limit_material_rates(time,completed_state,resolved_inlets,operating_conditions,limiter_dt=limiter_dt)

        # Everything outlet_values needs to report this point. The resolved
        # outlets are the ones the limiter has already throttled, so a unit
        # fed from this one receives what actually left.
        self._last_evaluation = {
            'time': time,
            'completed_state': completed_state,
            'resolved_inlets': resolved_inlets,
            'resolved_outlets': self._last_resolved_outlets,
            'operating_conditions': operating_conditions,
        }

        rates = self.sum_material_contributions(buffer.contributions)

        return rates, buffer

    def gather_inventory(self, completed_state):
        """Pack the current phase inventories into the material layout."""

        inventory = self._inventory_buffer

        for state_key, material_slice in self._positivity_groups:
            inventory[material_slice] = completed_state[state_key]

        return inventory

    @staticmethod
    def soft_saturate(ratio, width=0.1):
        """
        C1-continuous stand-in for ``min(ratio, 1)``.

        A hard minimum puts a kink in the right-hand side, so the integrator
        has to cut its step every time the limiter engages or disengages and
        the Jacobian it extrapolates from is wrong on one side of the
        switch. This version equals ``ratio`` below ``1 - width`` and ``1``
        above ``1 + width``, bridged by a quadratic whose value and slope
        match at both ends, so the derivative stays continuous.
        """

        lower = 1.0 - width
        upper = 1.0 + width

        blended = ratio - (ratio - lower) ** 2 / (4.0 * width)

        return np.where(
            ratio <= lower,
            ratio,
            np.where(ratio >= upper, 1.0, blended),
        )

    def calculate_scale(
            self,
            buffer,
            completed_state,
            limiter_dt=1.0,
        ):
        """
        Compute one throttling factor per phase inventory.

        Inlet and intraphase terms are treated as fixed; cross-phase and
        outlet terms compete for whatever inventory is left over
        ``limiter_dt`` and share a single factor per phase, which keeps an
        outlet composition consistent with the phase it drains.

        Returns ``None`` when nothing needs throttling, so the common case
        costs one vectorized comparison and no allocation of scale vectors.
        """

        contributions = buffer.contributions

        inventory = self.gather_inventory(completed_state)

        fixed = (
            contributions[buffer.INLET]
            + contributions[buffer.INTRAPHASE]
        )

        scalable = (
            contributions[buffer.CROSSPHASE]
            + contributions[buffer.OUTLET]
        )

        available = inventory + fixed * limiter_dt
        demand = -scalable * limiter_dt

        active = self._positivity_limited & (demand > 0.0)

        if not active.any():
            return None

        scales = self._scale_buffer
        scales.fill(1.0)

        scales[active] = self.soft_saturate(
            np.maximum(available[active], 0.0) / demand[active]
        )

        engaged = False

        for _, material_slice in self._positivity_groups:

            phase_scale = scales[material_slice].min()

            if phase_scale < 1.0:
                engaged = True

            scales[material_slice] = phase_scale

        if not engaged:
            return None

        return scales


    def limit_material_rates(
                self,
                time,
                completed_state,
                resolved_inlets,
                operating_conditions,
                limiter_dt=1.0,
            ):

            resolved_outlets = self._resolve_outlets(
                completed_state,
                operating_conditions
            )

            buffer = self.calculate_material_contributions(
                time,
                completed_state,
                resolved_inlets,
                resolved_outlets,
            )

            self.apply_rate_scaling(
                buffer,
                completed_state,
                limiter_dt
            )

            # Scaled in place above, so this is the outlet that actually left.
            self._last_resolved_outlets = resolved_outlets

            return buffer

    def apply_rate_scaling(
            self,
            buffer,
            completed_state,
            limiter_dt=1.0,
        ):
        """
        Throttle consuming contributions to keep phase inventories positive.

        ``limiter_dt`` is the horizon over which depletion is anticipated.
        Phases and nonlinear mechanisms are not re-evaluated: each mechanism
        that drains a throttled phase is scaled down at both ends, so the
        transfer stays closed even while it is being limited.

        Returns True when the limiter engaged.
        """

        scales = self.calculate_scale(
            buffer,
            completed_state,
            limiter_dt=limiter_dt,
        )

        if scales is None:
            return False

        self.scale_crossphase_transfers(buffer, scales)
        self.scale_outlet_transfers(buffer, scales)

        return True

    def _transfer_scale(self, scales, material_writes):
        """
        The factor a single mechanism may run at.

        A mechanism is held to the tightest limit among the phases it
        drains, so it can never take more from a phase than that phase's own
        factor allows.
        """

        factor = 1.0

        for material_slice, rate in material_writes:
            factor = min(factor, scales[material_slice.start])

        return factor

    def scale_crossphase_transfers(self, buffer, scales):
        """
        Scale each cross-phase mechanism down by a single factor.

        Both ends of a transfer move together, so throttling the source
        drain also throttles what arrives in the sink. Scaling the flat
        contribution rows independently would credit the sink with material
        the source never gave up.
        """

        row = buffer.contributions[buffer.CROSSPHASE]

        for aux in buffer.aux[buffer.CROSSPHASE]:

            material_writes = aux.get("material_writes")

            if not material_writes:
                continue

            factor = self._transfer_scale(scales, material_writes)

            if factor >= 1.0:
                continue

            for material_slice, rate in material_writes:
                row[material_slice] -= (1.0 - factor) * rate

    def scale_outlet_transfers(self, buffer, scales):
        """Scale outlet draws, and the streams that report them, together."""

        row = buffer.contributions[buffer.OUTLET]

        for transfer in buffer.aux[buffer.OUTLET]:

            material_slice = self._material_slice_by_phase.get(
                transfer.mapping.sink_phaseref
            )

            if material_slice is None:
                continue

            factor = scales[material_slice.start]

            if factor >= 1.0:
                continue

            row[material_slice] += (1.0 - factor) * transfer.species_flow

            for mechanism_slice, rate in transfer.material_writes:
                row[mechanism_slice] += (1.0 - factor) * rate

            transfer.scale(factor, self.basis)
            transfer.vol_flow *= factor

    def sum_material_contributions(self,contributions):
        return contributions.sum(axis=0)
    def resolve_outlet_flows(
        self,
        operating_conditions:dict[OperatingKey,Any],
    ):

        flows = {}
        for i, connection in enumerate(self.outlet_connections):
            key = OperatingKey("vol_flow",connection=i,port='outlet')
            if key in operating_conditions:
                flows[i] = operating_conditions[key]

        return flows
    
    def get_total_inlet_vol_flow(self,resolved_inlets:StreamConditions)->float:

        total = 0.0
        for inlet in resolved_inlets.streams:
            total += inlet.stream.vol_flow

        return total
    
    # Operating conditions that describe a whole connection rather than one
    # phase within it. A controller that asks for an outlet vol_flow means
    # the stream's total, which the caller distributes over the mapped
    # phases; handing the same number to every phase as its own flow would
    # multiply the draw by the number of phases.
    connection_level_operating_names = ("vol_flow",)

    def get_phase_operating_conditions(
        self,
        operating_conditions:dict[OperatingKey,Any],
        connection:int,
        phase_ref:PhaseRef,
        port:str,
        skip_connection_level=False,
    )->dict[OperatingKey,Any]:

        updates = {}
        port = port.lower()

        for key, value in operating_conditions.items():

            if key.connection != connection:
                continue

            if key.phaseref is None:
                if (skip_connection_level
                    and key.name in self.connection_level_operating_names):
                    continue
            elif key.phaseref != phase_ref:
                continue

            if (key.port is not None
                and key.port!=port):
                continue

            updates[key.name] = value

        return updates

    def compute_requested_phase_outlet_flow(
            self,
            vessel_phase,
            total_outlet_flow,
            connection,
        ):
        """Compute desired outlet flow for a vessel phase.

        The default implementation distributes the specified outlet
        connection flow among the mapped phases in proportion to their
        current volume within the vessel.

        If no total outlet flow is specified, no outlet flow is requested.
        """

        if total_outlet_flow is None:
            return 0.0

        total_vessel_flow = sum(
            self.phase_states.get_phase(m.sink_phaseref).vol
            for m in connection.phase_mappings)

        if total_vessel_flow <= 0:
            return 0.0

        fraction = vessel_phase.vol / total_vessel_flow

        return fraction * total_outlet_flow
    
    def compute_actual_phase_outlet_flow(
        self,
        vessel_phase,
        requested_flow,
    ):
        """
        Compute the physically achievable outlet flow for a vessel phase.

        The default implementation only rejects backflow. Draining a phase
        faster than it can supply is left to the positivity limiter, which
        throttles smoothly; clamping here would need a volumetric flow
        (m3/s) to be compared against a volume (m3), and the hard corner it
        introduced cost the integrator a step every time it was reached.

        Subclasses may override this to enforce additional constraints
        (e.g. settling, phase disengagement, hydraulics).
        """

        return max(requested_flow, 0.0)
    
    def _resolve_outlets(
        self,
        completed_state:dict,
        operating_conditions:dict[OperatingKey],
    )->StreamConditions:

        resolved = []

        outlet_flows = self.resolve_outlet_flows(operating_conditions)

        for connection_num, connection in enumerate(self.outlet_connections):# iterate over outlet streams

            outlet_stream = self._get_outlet_workspace(connection_num, connection)

            total_outlet_flow = outlet_flows.get(connection_num)
            transfers=[]
            for mapping in connection.phase_mappings: #iterate over each phase in that stream

                vessel_phase = self.phase_states.get_phase(mapping.sink_phaseref)
                outlet_phase = outlet_stream.get_phase_from_ref(mapping.source_phaseref)

                # Default outlet request
                requested_flow = self.compute_requested_phase_outlet_flow(vessel_phase,total_outlet_flow,connection)

                # Controller (or other operating conditions) may override the
                # request, but only a phase-specific vol_flow does: the
                # connection's total has already been distributed above.
                ops =self.get_phase_operating_conditions(
                    operating_conditions,
                    connection_num,
                    mapping.source_phaseref,
                    "outlet",
                    skip_connection_level=True,
                )
                requested_flow = ops.pop("vol_flow", requested_flow)

                # Apply physical limits once
                actual_flow = self.compute_actual_phase_outlet_flow(vessel_phase,requested_flow)

                # Only the intensive state is inherited from the vessel
                # phase: the amount is set by the resolved outlet flow, and
                # one composition basis fixes the rest. Reading the vessel
                # phase's full state_dict here evaluated every composition
                # representation just to discard all but one of them.
                composition_name = outlet_phase.default_composition_name

                updates = {
                    "temp": vessel_phase.temp,
                    "pres": vessel_phase.pres,
                    composition_name: getattr(vessel_phase, composition_name),
                }

                # Add any remaining operating-condition overrides
                updates.update(ops)

                # Physical limit always wins
                updates["vol_flow"] = actual_flow

                outlet_phase.updatePhase(**updates)
                species_flow = getattr(outlet_phase,self.basis+"_flow")
                transfers.append(
                    ResolvedPhaseTransfer(
                        connection=connection,
                        mapping=mapping,
                        vessel_phase=vessel_phase,
                        stream_phase=outlet_phase,
                        vol_flow=actual_flow,
                        species_flow=species_flow,
                        direction="outlet",
                    )
                )

            resolved.append(ResolvedStreamConnection(connection=connection,
                                                     transfers=transfers,
                                                     stream=outlet_stream))

        return StreamConditions(resolved)
        
    def resolve_inlets(self,completed_state,operating_conditions,inlet_inputs=None)->StreamConditions:

        resolved = []

        for connection_num, connection in enumerate(self.inlet_connections):

            inlet_stream = self._get_inlet_workspace(connection_num, connection)

            # The trajectory is recorded once per connection, so any
            # mapping entry carries the same connection-level states.
            trajectory = None

            for mapping in connection.phase_mappings:
                trajectory = (inlet_inputs or {}).get(
                    (connection_num, mapping.source_phaseref))

                if trajectory:
                    break

            phase_flows = self._inlet_phase_shares(connection, trajectory)

            transfers= []
            for mapping in connection.phase_mappings:
                stream_phase = inlet_stream.get_phase_from_ref(mapping.source_phaseref)

                try:
                    vessel_phase = self.phase_states.get_phase(
                        mapping.sink_phaseref)
                except KeyError:
                    # A slurry feed into a vessel with no matching phase.
                    # Without this it surfaced as a bare KeyError naming a
                    # PhaseRef, which says nothing about the flowsheet.
                    raise ValueError(
                        f'{type(self).__name__} is fed a '
                        f'{mapping.sink_phaseref.phase_type} phase but has '
                        'none to receive it. Give this vessel a '
                        f'{mapping.sink_phaseref.phase_type} phase, or put '
                        'a separation step upstream so it is not sent one.'
                    ) from None


                # As on the outlet side, a connection-level vol_flow
                # describes the whole stream. Handing it to each phase would
                # multiply a multi-phase feed by the number of phases.
                ops = self.get_phase_operating_conditions(
                    operating_conditions,
                    connection_num,
                    mapping.source_phaseref,
                    "inlet",
                    skip_connection_level=True,
                )

                workspace_key = (connection_num, mapping.source_phaseref)

                baseline = (inlet_inputs or {}).get(workspace_key)

                # Give this phase its share of the stream flow rather than
                # the whole of it, so the species_flow below is this phase
                # share too. The solid is deliberately left out: a vol_flow
                # on it drives set_mass, which would rescale the very
                # distribution just applied.
                if (phase_flows is not None and baseline
                        and 'vol_flow' in baseline
                        and mapping.source_phaseref.phase_type == 'liquid'):
                    baseline = dict(baseline)
                    baseline['vol_flow'] = phase_flows[
                        mapping.source_phaseref]

                self._apply_inlet_overrides(
                    stream_phase,
                    connection.stream.get_phase_from_ref(
                        mapping.source_phaseref
                    ),
                    ops,
                    workspace_key,
                    baseline=baseline,
                )


                species_flow = getattr(stream_phase,self.basis+"_flow")

                phase_vol_flow = stream_phase.vol_flow

                if phase_flows is not None:
                    phase_vol_flow = phase_flows[mapping.source_phaseref]


                transfers.append(
                    ResolvedPhaseTransfer(
                        connection=connection,
                        mapping=mapping,
                        vessel_phase=vessel_phase,
                        stream_phase=stream_phase,
                        vol_flow=phase_vol_flow,
                        species_flow=species_flow,
                        direction="inlet",
                    )
                )
            resolved.append(ResolvedStreamConnection(connection=connection,
                                                     transfers=transfers,
                                                     stream=inlet_stream))


        return StreamConditions(resolved)
    
    def calculate_material_contributions(
        self,
        time,
        completed_state,
        resolved_inlets,
        resolved_outlets,
    ):
        buffer = self._material_contributions
        buffer.reset()
        t0 = perf_counter()
        self.add_inlet_terms(
            buffer,
            time,
            completed_state,
            resolved_inlets,
            
        )
        self._timers['add_inlet_terms'] = self._timers.get('add_inlet_terms',0)+perf_counter()-t0

        t0 = perf_counter()
        self.add_intraphase_terms(
            buffer,
            time,
            completed_state,
        )
        self._timers['add_intraphase_terms'] = self._timers.get('add_intraphase_terms',0)+perf_counter()-t0
        t0 = perf_counter()
        self.add_crossphase_terms(
            buffer,
            time,
            completed_state,
        )
        self._timers['add_crossphase_terms'] = self._timers.get('add_crossphase_terms',0)+perf_counter()-t0

        t0 = perf_counter()
        self.add_outlet_terms(
            buffer,
            time,
            completed_state,
            resolved_outlets,
        )
        self._timers['add_outlet_terms'] = self._timers.get('add_outlet_terms',0)+perf_counter()-t0

        if self.dilutes_intensive_states:
            self.add_dilution_terms(
                buffer,
                completed_state,
                resolved_inlets,
                resolved_outlets,
            )
        return buffer

    def add_dilution_terms(
            self,
            buffer:MaterialContributionBuffer,
            completed_state,
            resolved_inlets,
            resolved_outlets,
        ):
        """Dilute per-volume mechanism states as the vessel volume changes.

        A population carried per m3 of slurry is diluted by every inflow,
        including a feed with no crystals, which never reaches the solid
        phase's inlet hook. The rate of volume change is taken as the
        difference between the volumetric inflow and outflow.
        """
        vol_in = sum(transfer.vol_flow or 0.0
                     for connection in resolved_inlets
                     for transfer in connection)
        vol_out = sum(transfer.vol_flow or 0.0
                      for connection in resolved_outlets
                      for transfer in connection)
        net_vol_flow = vol_in - vol_out

        if not net_vol_flow:
            return

        for phase in self.Phases.Phases:
            for mechanism in getattr(phase, 'mechanisms', None) or ():

                state_rates = mechanism.get_dilution_contributions(
                    net_vol_flow, completed_state)

                for state_key, value in state_rates.items():

                    state_slice = self.solver_state_collection.material_slices.get(state_key)

                    if state_slice is not None:
                        buffer.contributions[
                            buffer.INLET,
                            state_slice,
                        ] += value
    
    def add_inlet_terms(
            self,
            buffer:MaterialContributionBuffer,
            time,
            completed_state,
            resolved_inlets,
        ):
            for resolved_connection in resolved_inlets:
                for transfer in resolved_connection:

                    # Guarded the way add_outlet_terms already does it. A
                    # phase whose inventory is a mechanism state rather
                    # than species masses -- a solid tracked by its size
                    # distribution -- has no material slice, and indexing
                    # directly raised KeyError on the PhaseRef the moment
                    # a slurry was fed in. Its contribution arrives
                    # through the mechanism hook below instead.
                    material_slice = self._material_slice_by_phase.get(
                        transfer.mapping.sink_phaseref
                    )

                    if material_slice is not None:
                        buffer.contributions[
                            buffer.INLET,
                            material_slice,
                        ] += transfer.species_flow

                    # A distribution is carried per m3 of SLURRY (that is
                    # how MultiPhaseVessel.outputs publishes it, matching
                    # old PharmaPy), so the mechanism needs the whole
                    # stream's volumetric flow, not this phase's share of
                    # it. Using the solid's own vol_flow would understate
                    # the crystal feed by the solid volume fraction.
                    slurry_vol_flow = sum(
                        t.vol_flow or 0.0 for t in resolved_connection)

                    for mechanism in transfer.vessel_phase.mechanisms:

                        state_rates = mechanism.get_inlet_contributions(
                            transfer.stream_phase,
                            transfer.vessel_phase,
                            slurry_vol_flow,
                            completed_state,
                        )

                        for state_key, value in state_rates.items():

                            state_slice = self.solver_state_collection.material_slices.get(state_key)

                            if state_slice is not None:
                                buffer.contributions[
                                    buffer.INLET,
                                    state_slice,
                                ] += value

                    buffer.aux[
                        buffer.INLET
                    ].append(transfer)

        

    def add_outlet_terms(
        self,
        buffer:MaterialContributionBuffer,
        time,
        completed_state,
        resolved_outlets,
    ):

        for resolved_connection in resolved_outlets:
            for transfer in resolved_connection:

                material_slice = self._material_slice_by_phase.get(
                    transfer.mapping.sink_phaseref
                )
                if material_slice is not None:
                    buffer.contributions[
                        buffer.OUTLET,
                        material_slice,
                    ] -= transfer.species_flow

                for mechanism in transfer.vessel_phase.mechanisms:

                    state_rates = mechanism.get_outlet_contributions(
                        transfer.stream_phase,
                        transfer.vessel_phase,
                        transfer.vol_flow,
                        completed_state,
                    )

                    for state_key, value in state_rates.items():

                        state_slice = self.solver_state_collection.material_slices.get(state_key)

                        if state_slice is not None:
                            buffer.contributions[
                                buffer.OUTLET,
                                state_slice,
                            ] -= value

                            transfer.material_writes.append(
                                (state_slice, value)
                            )

                buffer.aux[
                    buffer.OUTLET
                ].append(transfer)

    
    def add_intraphase_terms(self,
            buffer:MaterialContributionBuffer,
            time,
            completed_state
        ):
        for process in self.intraphase_processes:
            phase = self.phase_states.get_phase(process.phaseref)

            intraphase_result = process.mechanism.get_solver_state_rates(
                process=process,
                phase=phase,
                time=time,
                completed_state=completed_state
            )

            for state_key,rate in intraphase_result.state_rates.items():
                state_slice = self.solver_state_collection.material_slices.get(state_key)
                if state_slice is not None:
                    buffer.contributions[
                        buffer.INTRAPHASE,
                        state_slice,
                    ] += rate

            buffer.aux[buffer.INTRAPHASE].append(intraphase_result.aux)

    def add_crossphase_terms(
            self,
            buffer:MaterialContributionBuffer,
            time:float,
            completed_state:dict[StateKey]
        ):

        # temp = completed_state["temp"]

        for connection in self.phase_connections:

            source_phase = self.phase_states.get_phase(connection.source_phaseref)

            sink_phase = self.phase_states.get_phase(connection.sink_phaseref)
            

            if not connection.active_condition(source_phase,sink_phase):
                continue

            t0 = perf_counter()
            crossphase_result = connection.mechanism.get_solver_state_rates(
                source_phase=source_phase,
                sink_phase=sink_phase,
                connection=connection,
                completed_state=completed_state,
                time=time
            )
            self._timers['crossphase_mechanism'] = self._timers.get('crossphase_mechanism',0)+perf_counter()-t0

            t0 = perf_counter()
            material_writes = []

            for state_key, rate in crossphase_result.state_rates.items():

                if isinstance(state_key.phaseref, BasePhase):
                    actual_phaseref = self.phase_states.get_ref(state_key.phaseref)

                    if actual_phaseref is None:
                        raise RuntimeError(
                            f"Could not find mechanism's phase: "
                            f"{state_key.phaseref} in vessel phases"
                        )

                    lookup_key = StateKey(state_key.name,actual_phaseref)
                else:
                    lookup_key = state_key

                material_slice = self.solver_state_collection.material_slices.get(lookup_key)

                if material_slice is not None:
                    buffer.contributions[
                        buffer.CROSSPHASE,
                        material_slice,
                    ] += rate

                    material_writes.append((material_slice, rate))

            self._timers['crossphase_contributions'] = self._timers.get('crossphase_contributions',0)+perf_counter()-t0

            crossphase_result.aux["material_writes"] = material_writes

            buffer.aux[buffer.CROSSPHASE].append(crossphase_result.aux)
    
    def energy_balances(
            self,
            time,
            completed_state,
            material_buffer
        ):
        """
        Energy contributions are accumulated in SI units (joules).

        Positive contributions add energy to the vessel.
        Negative contributions remove energy from the vessel.
        """

        aux = material_buffer.aux
        contributions = {
            "inlet": 0,
            "intraphase": 0,
            "crossphase": 0,
            "outlet": 0,
            "utility": 0,
            "mixing":0,
            "shaftwork":0
        }
        t0 = perf_counter()
        self.add_inlet_energy_terms(
            contributions,
            aux[material_buffer.INLET],
            time,
            completed_state
        )
        self._timers['add_inlet_energy_terms'] = self._timers.get('add_inlet_energy_terms',0)+perf_counter()-t0

        t0 = perf_counter()
        self.add_intraphase_energy_terms(
            contributions,
            aux[material_buffer.INTRAPHASE],
            time,
            completed_state
        )
        self._timers['add_intraphase_energy_terms'] = self._timers.get('add_intraphase_energy_terms',0)+perf_counter()-t0

        t0 = perf_counter()
        self.add_crossphase_energy_terms(
            contributions,
            aux[material_buffer.CROSSPHASE],
            time,
            completed_state
        )
        self._timers['add_crossphase_energy_terms'] = self._timers.get('add_crossphase_energy_terms',0)+perf_counter()-t0

        t0 = perf_counter()
        self.add_outlet_energy_terms(
            contributions,
            aux[material_buffer.OUTLET],
            time,
            completed_state
        )
        self._timers['add_outlet_energy_terms'] = self._timers.get('add_outlet_energy_terms',0)+perf_counter()-t0

        t0 = perf_counter()
        self.add_utility_energy_terms(
            contributions,
            time,
            completed_state
        )
        self._timers['add_utility_energy_terms'] = self._timers.get('add_utility_energy_terms',0)+perf_counter()-t0

        t0 = perf_counter()
        self.add_mixing_energy_terms(
            contributions,
            time,
            completed_state
        )
        self._timers['add_mixing_energy_terms'] = self._timers.get('add_mixing_energy_terms',0)+perf_counter()-t0

        t0 = perf_counter()
        self.add_shaftwork_energy_terms(
            contributions,
            time,
            completed_state
        )
        self._timers['add_shaftwork_energy_terms'] = self._timers.get('add_shaftwork_energy_terms',0)+perf_counter()-t0

        t0 = perf_counter()
        qdot = sum(contributions.values())
        self._timers['sum_energy_contributions'] = self._timers.get('sum_energy_contributions',0)+perf_counter()-t0
        t0 = perf_counter()
        basis = 'mass' if self.basis=='mass_j' else self.basis
        heat_capacity = self.Phases.getCp(basis = basis)
        self._timers['get_heat_capacity'] = self._timers.get('get_heat_capacity',0)+perf_counter()-t0
        dtemp_dt = qdot / heat_capacity
        return {StateKey("global_temp"): dtemp_dt}

    def add_inlet_energy_terms(
            self,
            contributions,
            aux:list[ResolvedPhaseTransfer],
            time,
            completed_state
        ):
        """Computes enthalpy effects"""
        for inlet in aux:

            phase = inlet.stream_phase

            h_in = phase.getEnthalpy(
                phase.temp,
                temp_ref=self.temp_ref,
                total_h=True,
                basis='mass'
            )  # J/kg mixture
            vessel_phase = inlet.vessel_phase

            h_vessel = vessel_phase.getEnthalpy(
                vessel_phase.temp,
                temp_ref=self.temp_ref,
                total_h=True,
                basis="mass",
            )

            contributions["inlet"] += (
                inlet.species_flow * (h_in - h_vessel)
            ).sum()

    def add_crossphase_energy_terms(
            self,
            contributions,
            aux,
            time,
            completed_state
        ):
        
        for crossphase in aux:
            connection = crossphase['connection']
            contributions['crossphase'] += connection.mechanism.get_heat_generation(aux=crossphase,
                                                             completed_state=completed_state,
                                                             time=time)
    def add_intraphase_energy_terms(self,
                                  contributions,
                                  aux,
                                  time,
                                  completed_state
                                ):
        for process_aux in aux:
            process = process_aux['process']
            q = process.mechanism.get_heat_generation(aux=process_aux,
                                                    completed_state=completed_state,
                                                    time=time)
            contributions['intraphase'] += q

    def add_outlet_energy_terms(
            self,
            contributions,
            aux:list[ResolvedPhaseTransfer],
            time,
            completed_state
        ):
        """Computes enthalpy effects.

        The balance is written for dT/dt, so -- like the inlet term -- each
        outlet enters relative to the vessel phase it is drawn from:
        -F_out * (h_out - h_vessel). A stream leaving at the vessel's own
        state carries no net heat; only a change of phase or temperature on
        the way out does. Subtracting the absolute F_out * h_out on top of
        the relative inlet term counted the outflow twice and pulled a
        continuous vessel towards temp_ref.
        """
        for outlet in aux:
            phase = outlet.stream_phase
            h_out = phase.getEnthalpy(
                phase.temp,
                temp_ref=self.temp_ref,
                total_h=True,
                basis='mass'
            )

            vessel_phase = outlet.vessel_phase
            h_vessel = vessel_phase.getEnthalpy(
                vessel_phase.temp,
                temp_ref=self.temp_ref,
                total_h=True,
                basis='mass'
            )

            contributions["outlet"] -= (
                outlet.species_flow * (h_out - h_vessel)
            ).sum()

    def get_heat_transfer_temperature(self):
        """The temperature to use to determine heat transfer from the vessel. The default assumption is to use the temperature of the first liquid phase"""
        return self.Phases[0].temp
    
    def add_utility_energy_terms(
            self,
            contributions,
            time,
            completed_state
        ):

        if not self.has_utility_balance:
            return

        temp = self.get_heat_transfer_temperature()
        temp_ht = self.Utility.temp_in#completed_state[StateKey("temp_ht")]

        qdot = self.get_heat_transfer_rate(temp, temp_ht)
                                           
        self.Utility.temp_out = temp_ht +qdot/(self.Utility.mass_flow*self.Utility.cp)

        contributions["utility"] -= qdot

    def add_mixing_energy_terms(self,contributions,time,completed_state):
        "Used to add heat of mixing terms"
        #should be += self.Phases.getHeatOfMixing()
        contributions['mixing']+=0

    def add_shaftwork_energy_terms(self,contributions,time,completed_state):
        contributions['shaftwork']+=0

    def get_heat_transfer_rate(
            self,
            temp,
            temp_ht
        ):

        if self.ht_mode == "coil":
            raise NotImplementedError

        area = self.get_heat_transfer_area()

        return self.u_ht * area * (temp - temp_ht)
    
        
    def get_heat_transfer_area(self):

        liquid = self.phase_states.get_phase(
            PhaseRef("liquid",0)
        )
        if self.diam <= 0:
            raise ValueError(
                "Heat transfer requires reactor diameter > 0."
            )

        return 4 * liquid.vol / self.diam + self.area_base
    
    
    
    def build_solver_history(
            self,
            time,
            solver_states
        ):

        history = self.solver_state_collection.unpack_history(
            solver_states
        )

        history["time"] = np.asarray(time)

        return history
    
    def update_final_state(self, solver_history):

        final_state = {
            key: value[-1]
            for key, value in solver_history.items()
            if key != "time"
        }

        self.update_phases_from_state(final_state)
        return final_state
    
    def update_final_conditions(self,completed_state,time,solver_history,output_history):
        completed_state = self.complete_state(completed_state,time[-1])
        # The final condition is one point in time, as the line above
        # already assumes. Passing the whole array made the controller and
        # the inlet trajectory return arrays where scalars belong.
        resolved_inlets,operating_conditions = self.get_operating_conditions(time[-1],completed_state)
        resolved_outlets = self._resolve_outlets(completed_state,operating_conditions)

        # The resolved streams live in a workspace that later evaluations
        # overwrite, so the reported final condition takes its own copy.
        self.outlet_conditions = copy.deepcopy(resolved_outlets)

        self.elapsed_time = time[-1]

        
    def retrieve_results(self, time, solver_states):

        solver_history = self.build_solver_history(time,solver_states)
        output_history = self.find_output_states_from_replay(time,solver_history)

        completed_state = self.update_final_state(solver_history)
        self.update_final_conditions(completed_state,time,solver_history,output_history)
        self.result = self.build_dynamic_result(time,
                    solver_history,
                    output_history)
        return self.result
    

    def build_dynamic_result(
        self,
        time,
        solver_history,
        output_history,
    ):

        data = {"time": np.asarray(time)}

        data.update(
            self.solver_state_collection.flatten(solver_history)
        )

        data.update(
            self.output_state_collection.flatten(output_history)
        )

        states_di = {}

        for key, state in self.solver_state_collection.states.items():

            states_di[
                self.solver_state_collection.format_key(key)
            ] = state.as_dict()

        fstates_di = {}

        for key, state in self.output_state_collection.states.items():

            fstates_di[
                self.output_state_collection.format_key(key)
            ] = state.as_dict()

        return DynamicResult(
            states_di,
            fstates_di,
            **data
        )
    @property
    def name_species(self):
        #Backward Compatibility
        return self.Phases.name_species

    @property
    def num_species(self):
        #backward compatibility
        return self.Phases.num_species
    def create_pseudo(self, memo=None):
        """
        Create an independent vessel for replaying an accepted solver
        trajectory.

        The pseudo vessel contains the model structure, phases, mechanisms,
        connections, and controller, but does not contain the integrator or
        other runtime solver objects.

        `memo` is a deepcopy memo to share with other pseudos. A
        ConcurrentVessel replays its members together and pre-registers an
        empty pseudo for each of them here, so a controller that watches
        another member is copied pointing at that member's pseudo -- which
        the joint replay keeps at the right state -- instead of at a frozen
        copy of the real vessel taken at the end of the run.
        """

        if memo is None:
            memo = {}

        pseudo = memo.get(id(self))

        if pseudo is None:
            pseudo = object.__new__(type(self))

            # Make self-references inside copied objects resolve to the
            # pseudo rather than accidentally retaining a reference to the
            # real vessel.
            memo[id(self)] = pseudo

        # Objects that are either non-copyable or represent runtime state
        # that should not be carried into replay.
        excluded = {
            "integrator",
            "result",
            "derivatives",
            "elapsed_time",
            "outlet_conditions",
            "_last_evaluation",
            "_last_resolved_outlets",
        }

        for name, value in self.__dict__.items():

            if name in excluded:
                continue

            pseudo.__dict__[name] = copy.deepcopy(
                value,
                memo,
            )

        # The pseudo is not an integratable vessel.
        pseudo.integrator = None
        pseudo._last_resolved_outlets = None

        # The phase lookup is keyed on id(phase), and those integer keys
        # survive the copy while the phases themselves are new objects, so
        # it has to be rebuilt against the pseudo's own phases. A standalone
        # replay never asked it a question; a joint replay evaluates the
        # material balances, and a cross-phase mechanism does.
        if pseudo.phase_states._compiled:
            pseudo.phase_states.compile(pseudo.Phases)

        pseudo.reset()

        return pseudo
    
    def find_output_states_from_replay(
        self,
        time,
        solver_history
    ):
        """
        Reconstruct all output states by replaying the accepted solver history.

        Parameters
        ----------
        solver_states : ndarray
            Solver state history returned by the integrator.

        time : ndarray
            Accepted/output time points corresponding to solver_states.

        Returns
        -------
        dict
            Dictionary keyed by StateKey containing the reconstructed output
            histories.
        """

        pseudo = self.create_pseudo()


        output_history = {
            key: []
            for key in pseudo.output_state_collection.states
        }

        for i, t in enumerate(time):

            # -----------------------------
            # Recover solver state
            # -----------------------------
            solver_state = {
                key: value[i]
                for key, value in solver_history.items()
                if key != "time"
            }

            outputs = pseudo.output_states_at(t, solver_state)

            for key, value in outputs.items():
                output_history[key].append(value)

        # Convert lists to arrays
        for key in output_history:

            output_history[key] = np.asarray(
                output_history[key]
            )

        return output_history

    def output_states_at(self, time, solver_state):
        """
        Every output state at one accepted point, as {StateKey: value}.

        One step of the replay, kept separate so that a ConcurrentVessel,
        which replays all of its members together, runs exactly this for
        each member at each point rather than a copy of it.
        """

        # -----------------------------
        # Reconstruct controller states
        # -----------------------------
        completed_state = self.complete_state(
            solver_state,
            time,
        )

        # -----------------------------
        # Update vessel phases
        # -----------------------------
        self.update_phases_from_state(completed_state)

        # -----------------------------
        # Reconstruct operating conditions
        # -----------------------------
        resolved_inlets, operating_conditions = (
            self.get_operating_conditions(time, completed_state))

        resolved_outlets = self._resolve_outlets(
            completed_state,
            operating_conditions,
        )

        # -----------------------------
        # Evaluate output variables
        # -----------------------------
        return {
            key: state.compute_value(
                state_var=state,
                time=time,
                completed_state=completed_state,
                context=self,
                resolved_inlets=resolved_inlets,
                resolved_outlets=resolved_outlets,
                operating_conditions=operating_conditions,
            )
            for key, state in self.output_state_collection.states.items()
        }
