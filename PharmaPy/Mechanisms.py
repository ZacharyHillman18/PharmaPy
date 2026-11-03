import warnings
from PharmaPy import xp as np
import copy
from PharmaPy.Distributions import (mass_fraction_to_number,
                                    to_extensive, to_intensive,
                                    volume_fraction_to_number)
from PharmaPy.DataClasses import (StateVariable,PhaseConnection,PhaseMapping,
                                  PhaseRef,PhaseStateCollection,PhaseStateVariable,
                                  StateKey,StateCollection,StreamConnection,IntraPhaseProcess,
                                  TransferResult,StateEvent
)
from PharmaPy.Phases_Refactored import BasePhase
from time import perf_counter

eps = np.finfo(float).eps

class Mechanism:
    """
    Base class for any mechanism that contributes material and/or energy.

    A mechanism may:
        * own internal differential states
        * own algebraic output states
        * compute material rates
        * compute heat generation

    The vessel knows nothing about the implementation.
    """

    # Window over which a mechanism that guards positivity should anticipate
    # depletion, in seconds. MultiPhaseVessel.compile_structure overwrites
    # this with its own positivity_horizon; the class default keeps a
    # mechanism usable on its own, outside a vessel.
    positivity_horizon = 1.0

    solver_states = ()
    output_states = ()
    owning_phase = None
    _solver_state_keys=()
    

    def __init__(self):
        self.exposed_attributes = set()
        self._timers = {}

    def _update_exposed_attributes(self):
        self.exposed_attributes.update(state.name for state in self.solver_states)

    def expose(self, *names):
        self.exposed_attributes.update(names)

    @property
    def solver_state_keys(self):
        return self._solver_state_keys
    def owns_state(self, state_key):
        """
        Return True if this mechanism owns this transported state.
        """
        return state_key in self.solver_state_keys
    def reset(self):
        """Optional."""
        pass

    def update_state(self, completed_state,**kwargs):
        t0 = perf_counter()

        # solver_states carries the template definitions, whose phaseref is
        # still None; the vessel stamps the owning phase onto the copies it
        # registers and records the resulting keys in solver_state_keys.
        # Rebuilding a key from the template here looks up
        # StateKey(name, None), which is never in completed_state, so the
        # mechanism silently kept its initial value for the whole solve.
        keys = self.solver_state_keys

        if not keys:
            keys = tuple(
                StateKey(variable.name, variable.phaseref)
                for variable in self.solver_states
            )

        for variable, key in zip(self.solver_states, keys):

            if key in completed_state:
                setattr(
                    self,
                    variable.name,
                    completed_state[key]
                )

        self._timers['update_state'] = self._timers.get('update_state',0)+perf_counter()-t0
    def add_solver_state_variables(
        self,
        collection,
        overwrite=False,
        phase_ref=None
    ):
        keys = []

        for state in self.solver_states:

            # A caller-supplied owner wins, but a state that already names
            # its own phase keeps it. Stamping phase_ref unconditionally
            # overwrote that with None for mechanisms attached to an
            # intraphase process, which registered the state under a key no
            # lookup could ever match.
            owner = phase_ref if phase_ref is not None else state.phaseref

            if owner is None:
                raise ValueError(
                    f"{type(self).__name__} defines solver states but has no "
                    "owning phase reference. Give the StateVariable a "
                    "phaseref, or attach the mechanism to a phase."
                )

            copiedState = copy.deepcopy(state)
            copiedState.phaseref = owner
            collection.add(copiedState, overwrite)
            keys.append(StateKey(state.name, owner))

        self._solver_state_keys = tuple(keys)

    def compile_solver_state_keys(self):
        self._solver_state_keys = tuple(
            StateKey(state.name, state.phaseref)
            for state in self.solver_states
        )

    def add_output_state_variables(
            self,
            state_collection,
            overwrite=False,
            **kwargs
        ):

        for state in self.output_states:
            state_collection.add(
                copy.deepcopy(state),
                overwrite
            )
    def get_solver_state_rates(self, **kwargs)->TransferResult:
        raise NotImplementedError

    def get_heat_generation(
        self,
        aux,
        completed_state,
        time
    ):
        return 0.0
    def get_solver_state_residuals(self, **kwargs):
        """
        Residuals for any solver state this mechanism declared as algebraic.

        Mirrors get_solver_state_rates, but returns {StateKey: residual}: the
        solver drives each residual to zero rather than integrating it. A
        mechanism with no algebraic states returns {}, and the vessel then
        skips the algebraic pass entirely.
        """
        return {}
    def get_overrides(self,name):
        return None
    def apply_inlet_state(self, name, value):
        """Set an inlet state this mechanism owns, from a trajectory.

        A crystal size distribution belongs to the mechanism rather than
        to the phase, so a feed carrying one cannot be applied through
        BasePhase.updatePhase. The vessel routes those states here.
        """
        setattr(self, name, np.asarray(value, dtype=float))

    def get_inlet_contributions(
        self,
        stream_phase,
        vessel_phase,
        amount,
        completed_state,
    ):
        return {}

    def get_outlet_contributions(
        self,
        vessel_phase,
        outlet_phase,
        amount,
        completed_state,
    ):
        return {}

    def get_dilution_contributions(self, net_vol_flow, completed_state):
        """Rate of change of an intensive state as the vessel volume changes.

        Only mechanisms whose state is a density per unit volume need it.
        """
        return {}

    def get_events(self,unit):
        return []

    # -----------------------------------------------------------------
    # Conversion to and from the pre-refactor phase representation.
    #
    # Old PharmaPy kept this state on the phase itself; here it lives on
    # the mechanism. A mechanism that predates the split therefore has to
    # say how to spell its state the old way, and how to read it back.
    # The base returns None for "I do not know how", which the phase turns
    # into an explicit error rather than a silently incomplete conversion --
    # a mechanism written after the split has no obligation to be
    # backwards compatible, but it must not pretend to be.
    # -----------------------------------------------------------------

    def to_legacy_state(self):
        """Constructor kwargs for the equivalent old-stack phase, or None."""
        return None

    @classmethod
    def from_legacy_phase(cls, legacy_phase, owning_phase):
        """Build this mechanism from an old phase, or return None."""
        return None

    @classmethod
    def claims_legacy_phase(cls, legacy_phase):
        """Whether this mechanism is the one that owns that phase's state."""
        return False
class CrossPhaseTransferMechanism(Mechanism):
    """
    Computes material exchanged between two phases.

    Returns
    -------
    transfer_rate : ndarray
        Species mass transfer rates.

    aux : dict
        Cached information required by get_heat_generation().
    """
    def __init__(self):
        super().__init__()
    def get_solver_state_rates(self,
            source_phase,
            sink_phase,
            connection,
            completed_state,
            time,
        )->TransferResult:

        raise NotImplementedError
    def update_outlet(self, outlet_phase, completed_state, operating_conditions):
        pass
    def update_inlet(self, inlet_phase, completed_state, operating_conditions):
        pass
class MechanismView:

    def __init__(self, mechanism, direction):
        self.mechanism = mechanism
        self.direction = direction

    def __getattr__(self, name):
        mechanism = object.__getattribute__(self, 'mechanism')
        return getattr(mechanism, name)
    def __deepcopy__(self, memo):
        cls = type(self)
        copied = cls.__new__(cls)
        memo[id(self)] = copied

        copied.mechanism = copy.deepcopy(self.mechanism, memo)
        copied.direction = self.direction

        return copied

    def get_solver_state_rates(
            self,
            *args,
            **kwargs,
        ):

        result = self.mechanism.get_solver_state_rates(
            *args,
            **kwargs,
        )

        # Each view keeps the result only on its own side of zero, and the
        # forward (crystallization) view owns zero itself. When both kept it,
        # a net rate of exactly zero counted the population balance twice:
        # a moments crystallizer (nucleus size 0) starts with mu_2 = 0, so its
        # net rate is zero while nucleation is not, and mu_0 grew at twice
        # the rate there. The original method_of_moments evaluates once.
        if np.is_traced(result.net_mass_rate):
            # A trace multiplies by the test rather than dropping the rates,
            # so the result has the same keys whichever side it is on.
            if self.direction == "forward":
                keep = result.net_mass_rate >= 0
            else:
                keep = result.net_mass_rate < 0

            return TransferResult(
                state_rates={key: np.where(keep, rate, 0.0)
                             for key, rate in result.state_rates.items()},
                aux=result.aux,
                net_mass_rate=np.where(keep, result.net_mass_rate, 0.0),
            )

        if self.direction == "forward":
            if result.net_mass_rate < 0:
                return self.zero_out(result)

        else:
            if result.net_mass_rate >= 0:
                return self.zero_out(result)

        return result
    def zero_out(self,result:TransferResult):
        return TransferResult(
            state_rates={},
            aux=result.aux,
            net_mass_rate=0.0,
        )
    def get_heat_generation(self, *args, **kwargs):

        if self.direction == "reverse":
            return getattr(self.mechanism.mechanism_kinetics,'heat_dissol',0.0)

        return self.mechanism.get_heat_generation(
            *args,
            **kwargs,
        )

    
class ReversibleTransferMechanism:

    def __init__(
        self,
        source_mechanism:CrossPhaseTransferMechanism,
    ):

        self.forward = MechanismView(source_mechanism,"forward")
        self.reverse  = MechanismView(source_mechanism,"reverse")

    
class ReactionMechanism(Mechanism):
    """
    Computes intraphase material generation or consumption.

    Returns
    -------
    species_mass_rates : ndarray
        Species generation/consumption rates.

    aux : dict
        Cached reaction information required by
        get_heat_generation().
    """
    def __init__(
        self,
        kinetics,
        owning_phase,
        molarity_in_L=True
    ):
        super().__init__()
        self.kinetics = kinetics
        self.molarity_in_L = molarity_in_L
        self.owning_phase=owning_phase
    def get_solver_state_rates(
            self,
            process:IntraPhaseProcess,
            phase,
            time,
            completed_state
        )->TransferResult:
        "aux must have process field that stores process"
        # mole_adjust = 1000 if self.molarity_in_L else 1
        

        temp = phase.temp

        mask = np.array([
            species in self.kinetics.partic_species
            for species in phase.name_species
        ])

        conc = np.maximum(phase.mole_conc,0.0) #sanitize input incase integrator gave slightly negative
        deltah_rxn = None

        if self.kinetics.keq_params is not None:
            deltah_rxn = phase.getHeatOfRxn(
                self.kinetics.stoich_matrix,
                temp,
                mask,
                self.kinetics.delta_hrxn,
                self.kinetics.tref_hrxn
            )


        reaction_rates,species_rates = self.kinetics.get_rxn_rates(
            conc[mask],
            temp,
            return_both=True,
            delta_hrxn = deltah_rxn,
            # The same window the vessel's own positivity limiter uses, so
            # the two guards agree on how far ahead they are looking.
            limiter_dt=self.positivity_horizon,
        )

        species_massPerVol_rates = np.zeros(phase.num_species)
        species_massPerVol_rates = np.set_at(species_massPerVol_rates, mask, species_rates)
        species_massPerVol_rates *= phase.mw
        species_mass_rates = species_massPerVol_rates* phase.vol   # kmol/(m3 s) * kg/kmol * m3 = kg/s
        state_rates = {StateKey('mass_j',process.phaseref):species_mass_rates}
        aux = {
                "phase": phase,
                "rxn_rates": reaction_rates,
                "process":process
                }
        result = TransferResult(state_rates=state_rates,aux=aux,net_mass_rate=species_mass_rates.sum())
        return result
    
    def get_heat_generation(self, aux, completed_state, time):
        phase = aux['phase']
        temp = phase.temp
        rk = self.kinetics
        mask = np.array([
            species in rk.partic_species
            for species in phase.name_species
        ])

        deltah_rxn = (
            phase.getHeatOfRxn(
                rk.stoich_matrix,
                temp,
                mask,
                rk.delta_hrxn,
                rk.tref_hrxn
            ))
        # Rates are in kmol/(m3 s) (mole_conc is kmol/m3 = mol/L) and the
        # volume in m3, so rate*vol is kmol/s. deltah_rxn is J/mol, hence the
        # 1000 mol/kmol to get W. Negative deltah_rxn (exothermic) heats.
        q = -(deltah_rxn * aux["rxn_rates"]).sum() * phase.vol * 1000
        return q
    
    


   
    

    
class DirectTransfer(CrossPhaseTransferMechanism):
    """
    Move species from a connection's source phase to its sink at a rate the
    mechanism owns.

    ``rate`` is the mass flow from source to sink, in kg/s:

    * a callable, ``rate(source_phase, sink_phase, time, completed_state)``;
    * or a constant.

    An array gives the flow of each species, and a negative entry flows the
    other way. A scalar is a total, split across species by the connection's
    ``species_weights`` when it has them, otherwise in the proportions of the
    source phase's own composition (the source gives material up as it is).

    ``heat_of_transfer`` is the enthalpy change of the material moved, in J
    per kg moved from source to sink (sink minus source, so negative when
    the transfer releases heat, as a condensation does). The energy balance
    receives -heat_of_transfer times the net mass moved.

    The vessel's positivity limiter scales this transfer like any other, so
    a rate that would empty the source is throttled rather than driving it
    negative. On a traced backend a callable rate runs under the tracer, so
    it must follow the PharmaPy.xp rules.

    The phases' material states must be on a mass basis (``mass_j``, the
    vessel default).
    """

    def __init__(self, rate, heat_of_transfer=0.0):
        super().__init__()
        self.rate = rate
        self.heat_of_transfer = heat_of_transfer

    @classmethod
    def first_order(cls, rate_constant, species=None, heat_of_transfer=0.0):
        """
        Transfer at ``rate_constant`` (1/s) times the source's mass of each
        species. ``species`` names the species that move; None moves all.
        """

        def rate(source_phase, sink_phase, time, completed_state):
            flows = rate_constant * source_phase.mass_j

            if species is not None:
                moving = np.array([name in species
                                   for name in source_phase.name_species],
                                  dtype=float)
                flows = flows * moving

            return flows

        return cls(rate, heat_of_transfer)

    def species_flows(self, source_phase, sink_phase, connection,
                      completed_state, time):
        """kg/s of each species, from source to sink."""

        if callable(self.rate):
            flows = self.rate(source_phase, sink_phase, time, completed_state)
        else:
            flows = self.rate

        flows = np.asarray(flows, dtype=float)

        if flows.ndim == 0:
            weights = connection.species_weights

            if weights is None:
                weights = source_phase.mass_frac

            flows = flows * np.asarray(weights, dtype=float)

        return flows

    def get_solver_state_rates(
            self,
            source_phase,
            sink_phase,
            connection,
            completed_state,
            time,
        ) -> TransferResult:

        source_key = StateKey('mass_j', connection.source_phaseref)
        sink_key = StateKey('mass_j', connection.sink_phaseref)

        if source_key not in completed_state or sink_key not in completed_state:
            raise ValueError(
                "DirectTransfer moves mass between phases whose material "
                f"states are mass_j; {source_key} or {sink_key} is not a "
                "state of this vessel.")

        flows = self.species_flows(source_phase, sink_phase, connection,
                                   completed_state, time)
        net = np.sum(flows)

        return TransferResult(
            state_rates={source_key: -flows, sink_key: flows},
            aux={
                "connection": connection,
                "species_flows": flows,
                "net_mass_rate": net,
            },
            net_mass_rate=net,
        )

    def get_heat_generation(self, aux, completed_state, time):
        return -self.heat_of_transfer * aux["net_mass_rate"]

class PopulationBalanceMechanism(CrossPhaseTransferMechanism):
    """
    Base class for crystallization population balance mechanisms.

    Responsibilities
    ----------------
    * Evaluate crystallization kinetics.
    * Compute supersaturation/solubility.
    * Compute distribution moments.
    * Convert crystal growth into species transfer rates.
    * Define common output variables.

    Child classes are responsible only for solving the population
    balance equation (FVM, MOM, QMOM, etc.).
    """

    def __init__(
        self,
        owning_phase:BasePhase,
        target_components:str|list,
        solvent_name:str,
        kinetics=None,
        density=None,
        kv=1,
        fraction=None
    ):
        """
        owning_phase: BasePhase the instance of the phase that owns the population balance (e.g. the solid phase for a crystallization)
            the owning_phase.mass_frac should be the mass_frac of the resulting members. for multiple mass_frac
            you will need multiple mechanisms
        kinetics: the kinetics object that describes how the mechanism takes place. If it is not present, 
            the mechanism only exists to track state properties not dynamics
        kv: float the volumetric shape factor
            
        """
        super().__init__()
        self._mechanism_kinetics = kinetics
        self.owning_phase = owning_phase
        if isinstance(target_components, str):
            target_components = [target_components]
        self.target_components = target_components
        self.solvent_name =solvent_name
        
        self._density = density

        if self.target_components is not None:
            self.target_ind = []
            for tc in self.target_components:
                name_bool = [name == tc for name in self.owning_phase.name_species] #TODO check that it selects correctly
                self.target_ind.append(np.where(name_bool)[0][0])
        self.solvent_ind = self.owning_phase.name_species.index(self.solvent_name)

        
        self.kv = kv
        self.output_states=[StateVariable(name="supersat",dim=len(self.target_ind),units="-",state_type="post", compute_value=self.compute_supersat_output),
                    StateVariable(name="solubility",dim=len(self.target_ind),units="kg/m3",state_type="post",compute_value=self.compute_solubility_output),
                    StateVariable(name="mu_n",dim=4,index=[0, 1, 2, 3],units="m**n",state_type="post", compute_value=self.compute_moments_output)]
        if fraction is None:
            fraction = np.zeros(self.owning_phase.num_species)
            fraction[self.target_ind] = np.full(len(self.target_components),1/len(self.target_components))
        self.fraction = fraction

        # Set on every right-hand side evaluation; see _slurry_volume.
        self.slurry_volume = None

        # The liquid this population is suspended in, set when the vessel
        # wires its phase connections. Needed to turn an intensive population
        # into an inventory at any point, not just after a solve.
        self.liquid_phase = None

        self._timers = {}

    @property
    def mechanism_kinetics(self):
        if self._mechanism_kinetics is not None:
            return self._mechanism_kinetics
        raise AttributeError("No kinetics were specified")
    
    @mechanism_kinetics.setter
    def mechanism_kinetics(self,value):
        if value is not None:
            self._mechanism_kinetics = value
            

    def getDensity(self):
        return self.density
    
    @property
    def density(self):
        if self._density is not None:
            return self._density
        return self.owning_phase.density
    
    def get_overrides(self):
        overrides = {
            "mass": self.get_mass,
            "set_mass": self.set_mass,
        }

        if self._density is not None:
            overrides["getDensity"] = self.getDensity

        return overrides
    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    # x_grid is in microns. Moments are reported in m**n, one factor of this
    # per moment order, matching SolidPhase.getMoments.
    MOMENT_UNIT_FACTOR = 1e-6

    def compute_moments(
        self,
        distrib,
        x_grid,
    ):
        """
        Compute moments from a number density distribution, in m**n.

        ``x_grid`` is in microns, so the raw integral is in micron**n. The
        conversion to metres is what SolidPhase.getMoments applies, and the
        original crystallizer relies on it: its kinetics and its
        ``vol_solid = mu_n[3] * kv`` both expect m**n. Returning the raw
        micron-based integral instead silently rescaled mu_1 by 1e6, mu_2 by
        1e12 and mu_3 by 1e18.

        Use compute_second_moment / compute_third_moment where a micron-based
        integral is wanted instead.
        """

        moments = np.zeros(4)

        for n in range(4):
            moments = np.set_at(moments, n,
                                self.integrate_over_size(distrib * x_grid**n,
                                                         x_grid))

        return moments * self.MOMENT_UNIT_FACTOR ** np.arange(4)

    def integrate_over_size(self, integrand, x_grid=None):
        """Quadrature for a moment integral.

        A scheme that carries cell averages must integrate them the same way
        it advances them, or its moments do not obey its own conservation
        law. Subclasses override this; the generic rule is the trapezoid.
        """
        return np.trapezoid(integrand, self.x_grid if x_grid is None
                            else x_grid)

    
    def liquid_in_solubility_basis(self, liquid):
        """Liquid composition converted into the kinetics' declared basis.

        A solubility correlation is just numbers; which composition basis it
        was fitted in lives only in the person who fitted it. CrystKinetics
        now states it, and this converts the liquid to match, so the
        supersaturation and the solubility are always compared like for like.

        Previously this was hard-coded to mass per volume of *pure solvent*
        while original PharmaPy feeds solubility_temp a mass concentration on
        a *solution* basis -- a silent ~5% offset on concentration that the
        exponents amplified to ~13% on growth and ~31% on nucleation.
        """
        basis = getattr(self.mechanism_kinetics, 'solubility_basis',
                        'mass_per_volume_solution')

        if basis == 'mass_per_volume_solution':
            return liquid.mass_conc

        if basis == 'mole_per_volume_solution':
            return liquid.mole_conc

        if basis == 'mass_per_mass_solution':
            return liquid.mass_frac

        if basis == 'mole_frac':
            return liquid.mole_frac

        mass_solvent = liquid.mass_j[self.solvent_ind]

        if basis == 'mass_per_mass_solvent':
            return liquid.mass_j / mass_solvent

        if basis == 'mass_per_volume_solvent':
            return (liquid.mass_j / mass_solvent
                    * liquid.getDensityPure()[0][self.solvent_ind])

        raise ValueError(
            f"Unknown solubility basis {basis!r}. Valid bases are "
            f"{list(getattr(self.mechanism_kinetics, 'SOLUBILITY_BASES', ()))}"
        )
    def compute_supersaturation(
        self,
        liquid,
    ):
        "Express the liquid in whatever basis the solubility is stated in."
        conc = self.liquid_in_solubility_basis(liquid)

        # An index array rather than the list: same selection, and a traced
        # array accepts only the array form.
        conc_target = conc.T[np.asarray(self.target_ind)]
        
        # Supersaturation
        solubility = self.mechanism_kinetics.get_solubility(liquid.temp, conc)
        supersat = (conc_target - solubility)

        if self.mechanism_kinetics.sup_sat_type == 'relative':
            supersat = supersat / solubility

        if self.mechanism_kinetics.sup_sat_type == 'ratio':
            supersat = supersat / solubility + 1
        
        return conc, supersat, solubility

    # ------------------------------------------------------------------
    # Crystal -> liquid coupling
    # ------------------------------------------------------------------

    def compute_species_transfer(
        self,
        crystal_mass_rate,
        liquid,
    ):
        """
        Convert crystal growth (kg crystal/s)
        into species transfer.

        Positive crystal growth removes material
        from the liquid.
        """

        rates = np.zeros(liquid.num_species)

        mask = self.target_ind

        # Assume equal split unless overridden later.
        rates = np.set_at(rates, np.asarray(mask),
                          -crystal_mass_rate / len(mask))

        return rates

    # ------------------------------------------------------------------
    # API for child classes
    # ------------------------------------------------------------------

    def solve_population_balance(
        self,
        liquid:BasePhase,
        solid:BasePhase,
        completed_state:dict[StateKey],
        time:float,
    )->TransferResult:
        """
        Child classes implement this.

        Returns
        -------
        distribution_rate
        crystal_mass_rate
        aux
        """

        raise NotImplementedError

    # ------------------------------------------------------------------
    # Vessel interface
    # ------------------------------------------------------------------

    def get_solver_state_rates(
        self,
        source_phase:BasePhase,
        sink_phase:BasePhase,
        connection:PhaseConnection,
        completed_state:dict[StateKey],
        time:float,
    )->tuple[dict[StateKey],dict]:
        """
        Called by the vessel.

        Crystallization runs liquid -> solid, but dissolution is wired as a
        second connection running solid -> liquid, so source and sink swap
        roles between the two. The population balance is written in terms of
        liquid and solid, not source and sink, so identify them by phase
        family rather than by connection direction. Reading them positionally
        bound the solid to the liquid slot on the dissolution connection,
        where its all-zero mass_j produced 0/0 and poisoned the whole
        right-hand side with NaN.
        """
        liquid_phase, solid_phase = self._orient_phases(source_phase,
                                                        sink_phase)

        t0 = perf_counter()
        result= self.solve_population_balance(liquid_phase,solid_phase,completed_state,time,connection)
        self._timers['pop_balance_solve_total'] = self._timers.get('pop_balance_solve_total',0)+perf_counter()-t0
        if not hasattr(self,"liquid_phase_ref"):
            self.liquid_phase_ref = self._liquid_phaseref(connection)

        result.aux.update(
            {
                "connection": connection,
                "state_rates": result.state_rates,
                "liquid_phase": liquid_phase,
                "solid_phase": solid_phase,
            }
        )
        return result

    @staticmethod
    def _orient_phases(source_phase, sink_phase):
        """Return (liquid, solid) regardless of connection direction."""
        if getattr(source_phase, 'phase_family', None) == 'solid':
            return sink_phase, source_phase

        return source_phase, sink_phase

    def _liquid_phaseref(self, connection):
        """The PhaseRef naming the liquid side of this connection."""
        if connection.source_phaseref.phase_type == 'solid':
            return connection.sink_phaseref

        return connection.source_phaseref

    # ------------------------------------------------------------------
    # Heat generation
    # ------------------------------------------------------------------

    def get_heat_generation(
        self,
        aux,
        completed_state,
        time,
    ):
        """
        Heat of crystallization.

        Default assumes none.
        """

        return 0.0

    def get_events(self, unit):
        """
        Crystallization changes regime when supersaturation changes sign:
        growth and primary nucleation above zero, dissolution below. The
        kinetics branch there, so the right-hand side is only piecewise
        smooth, and the integrator has to be told where the corner is instead
        of hunting for it with short steps.
        """

        if self._mechanism_kinetics is None:
            return []

        liquid_ref = getattr(self, "liquid_phase_ref", None)

        if liquid_ref is None:
            return []

        def supersaturation_sign(time, completed_state, unit):
            liquid = unit.Phases.get_phase_from_ref(liquid_ref)
            supersat = np.atleast_1d(self.compute_supersaturation(liquid)[1])
            return float(supersat[0])

        return [
            StateEvent(
                name=f"supersaturation_sign[{type(self).__name__}]",
                function=supersaturation_sign,
                direction=0,
                terminal=False,
                source=self,
            )
        ]

    def compute_supersat_output(
        self,
        state_var,
        time,
        completed_state,
        context,
        resolved_inlets=None,
        resolved_outlets=None,
        operating_conditions=None,
    ):
        """
        Compute supersaturation from the current replay state.

        This is the StateVariable.compute_value interface. The mechanism
        itself is the context during replay.
        """

        liquid = context.Phases.get_phase_from_ref(
            self.liquid_phase_ref
        )


        return self.compute_supersaturation(liquid)[1]

    def compute_solubility_output(
        self,
        state_var,
        time,
        completed_state,
        context,
        resolved_inlets=None,
        resolved_outlets=None,
        operating_conditions=None,
    ):
        """
        Compute solubility from the current replay state.
        """

        liquid = context.Phases.get_phase_from_ref(
            self.liquid_phase_ref
        )

        return self.compute_supersaturation(liquid)[2]

    def compute_moments_output(
        self,
        state_var,
        time,
        completed_state,
        context,
        resolved_inlets=None,
        resolved_outlets=None,
        operating_conditions=None,
    ):
        distribution = completed_state[
            StateKey(
                self.distribution_state_name,
                self.owning_phase_ref,
            )
        ]

        return self.compute_moments(
            distribution,
            self.x_grid,
        )
    
    # ---------------- basis helpers ----------------
    #
    # The population state is a number DENSITY per m3 of slurry. The slurry
    # volume therefore appears at exactly two boundaries, where an intensive
    # quantity becomes an extensive one: the mass transfer leaving the
    # population balance, and the solid inventory. Nowhere else -- in
    # particular NOT on the nucleation rate, which is already #/(s m3 slurry).
    #
    # The old `scale` argument was meant as numerical conditioning, but it
    # multiplied the state without being divided back out of the moments, so
    # it acted as a nucleation multiplier and moved a batch answer by 1e8
    # between scale=1 and scale=1e-9. It is accepted and ignored now; solver
    # tolerances do this job.

    @staticmethod
    def _reject_scale(scale):
        if scale is None or scale == 1:
            return

        warnings.warn(
            'scale= is deprecated and ignored. It was meant as numerical '
            'conditioning but leaked into the physics, acting as a '
            'nucleation multiplier. The population balance is now written so '
            'that no conditioning factor is needed.',
            DeprecationWarning,
            stacklevel=3,
        )

    def true_state(self, state):
        """Kept so callers written against the scaled state still work."""
        return np.asarray(state, dtype=float)

    def slurry_volume_from(self, liquid, true_state):
        """Slurry volume implied by the liquid volume and the solid fraction.

        Under the intensive basis kv * mu_3 is a volume FRACTION (m3 of
        crystal per m3 of slurry), not a volume, so the old
        ``vol_liquid + kv*mu_3`` no longer makes sense dimensionally:
        vol_slurry = vol_liquid / (1 - phi).

        Takes the TRUE (unscaled) population state and asks
        solid_volume_fraction for phi, so there is one convention for the
        fraction rather than one helper wanting micron-based moments and
        another metre-based ones.
        """
        phi = self.solid_volume_fraction(true_state)

        # Read once: the liquid's volume is its mass over a mixture density.
        vol_liquid = liquid.vol

        if np.is_traced((phi, vol_liquid)):
            # The same clamp and division as below, without asking Python
            # for the numbers.
            vol_slurry = vol_liquid / (1.0 - np.clip(phi, 0.0, 1.0 - 1e-12))
        else:
            phi = min(max(phi, 0.0), 1.0 - 1e-12)
            vol_slurry = float(vol_liquid) / (1.0 - phi)

        self.slurry_volume = vol_slurry

        return vol_slurry

    # ---------------- extensive <-> intensive ----------------
    #
    # Intensive is what this stack stores: the population state is a number
    # density per m3 of slurry. Extensive -- an absolute count for the whole
    # phase -- is an input and output format, and these two methods are the
    # only sanctioned way across that boundary.

    def to_intensive(self, distrib, vol_slurry=None):
        """
        Absolute counts to this stack's per-m3-of-slurry number density.

        ``vol_slurry`` defaults to the mechanism's own, which exists once it
        is attached to a phase inside a vessel. A solid on its own has no
        slurry volume - that is why its mass reads zero - so pass the value
        explicitly in that case rather than getting a silent factor wrong.
        """

        return to_intensive(distrib, self._resolve_slurry_volume(vol_slurry))

    def to_extensive(self, distrib=None, vol_slurry=None):
        """
        The stored number density back to absolute counts, for a consumer
        that wants an extensive distribution.
        """

        if distrib is None:
            distrib = getattr(self, self.solver_states[0].name)

        return to_extensive(distrib, self._resolve_slurry_volume(vol_slurry))

    def _resolve_slurry_volume(self, vol_slurry=None):
        """The conversion factor, from the argument or from the phase."""

        if vol_slurry is not None:
            return float(vol_slurry)

        volume = self._slurry_volume()

        if not volume:
            raise ValueError(
                "No slurry volume is available for an extensive/intensive "
                "conversion. The slurry volume is the liquid's, divided by "
                "(1 - solid volume fraction), so a solid that is not yet "
                "part of a vessel does not have one. Pass vol_slurry "
                "explicitly."
            )

        return float(volume)

    def _slurry_volume(self, true_state=None):
        """Slurry volume for the inventory accessors.

        The inventory is read DURING a right-hand side evaluation -- the
        solid's vol setter calls set_mass -- and that can happen before
        solve_population_balance has run and cached a volume. So compute it
        from the liquid phase when one is known, and only fall back to the
        cache. Raising here instead surfaced as an opaque
        'repeated recoverable right-hand side errors' from CVode.
        """
        liquid = getattr(self, 'liquid_phase', None)

        if liquid is not None and true_state is not None:
            try:
                return self.slurry_volume_from(liquid, true_state)
            except (AttributeError, TypeError, ValueError):
                # A trace that fails here must say so. Falling back to the
                # cache would quietly freeze the slurry volume into the
                # compiled right-hand side.
                if np.is_traced(true_state):
                    raise

        if self.slurry_volume is not None:
            return self.slurry_volume

        return 0.0

    # ---------------- flow terms ----------------
    #
    # MultiPhaseVessel calls these for every resolved transfer whose
    # vessel-side phase owns this mechanism. Neither was implemented, so
    # both fell through to Mechanism's empty dict and crystals never
    # entered through a feed nor left with the product: a continuous
    # crystallizer was a batch crystallizer with liquid flowing past it.
    #
    # Reference is old MSMPR.material_balances (Crystallizers.py), which
    # writes the same expression for both discretisations:
    #
    #     tau_inv    = input_flow / vol
    #     flow_state = tau_inv * (input_state - state)
    #
    # for a per-slurry-volume population n, which is exactly the basis
    # these mechanisms carry, so it applies unchanged:
    #
    #     dn/dt = (Q / vol) * n_in  -  (Q / vol) * n
    #
    # the first term being the inlet half and the second the outlet half.
    #
    # SIGN CONVENTION: both hooks return a positive magnitude. The vessel
    # adds inlet contributions and SUBTRACTS outlet ones
    # (MultiPhaseVessel.add_outlet_terms does '-= value', matching how it
    # treats species_flow), so returning a negative rate here would feed
    # crystals back in rather than withdrawing them.
    #
    # Both terms are exactly zero for a vessel with no connections, which
    # keeps a batch crystallizer untouched. They live on the base class so
    # a resolved distribution and a set of moments behave identically --
    # the moments path having silently had no flow terms at all.

    # Populations are carried in micron-based units, so a third moment
    # needs this to become m3.
    VOLUME_UNIT_FACTOR = 1e-18

    @property
    def flow_state_name(self):
        """The population state that travels with the stream."""
        for attr in ('distribution_state_name', 'moments_state_name'):

            name = getattr(self, attr, None)

            if name:
                return name

        return None

    def _flow_state_key(self):
        return StateKey(self.flow_state_name, self.owning_phase_ref)

    def _validate_inlet_state(self, value):
        """Hook for a subclass to reject a feed it cannot accept."""
        return value

    def solid_volume_fraction(self, value):
        """Solid volume per m3 of slurry, from a per-slurry-volume state.

        Used to split a slurry feed's volumetric flow between its phases.
        """
        raise NotImplementedError

    def _inlet_population_density(self, stream_phase):
        """Feed population per m3 of slurry.

        Published in that basis by MultiPhaseVessel.outputs, which is what
        old PharmaPy moved between units -- the old crystallizer
        multiplied by vol_slurry again on the way back into a phase.
        """
        name = self.flow_state_name

        for mechanism in getattr(stream_phase, 'mechanisms', None) or ():

            if getattr(mechanism, 'flow_state_name', None) != name:
                continue

            value = getattr(mechanism, name, None)

            if value is None:
                continue

            return self._validate_inlet_state(
                np.asarray(value, dtype=float))

        return None

    def get_inlet_contributions(self, stream_phase, vessel_phase, amount,
                                completed_state):

        if not np.is_traced(amount) and not amount:
            return {}

        density = self._inlet_population_density(stream_phase)

        if density is None:
            return {}

        # 'amount' is the WHOLE slurry flow of the feed (MultiPhaseVessel.
        # add_inlet_terms), so it is divided by the vessel's SLURRY volume,
        # not by the volume of the solid phase that owns this mechanism (the
        # crystal volume). Dividing by the latter overstated the crystal
        # inflow by 1/phi, and gave none at all into a vessel without
        # crystals yet. Intensive state, so this is the MSMPR form directly:
        #     dn/dt = (Q/V_slurry) * (n_in - n)
        key = self._flow_state_key()
        state = completed_state.get(key)

        if state is None:
            state = getattr(self, self.flow_state_name)

        vol_slurry = self._slurry_volume(
            self.true_state(np.asarray(state, dtype=float)))

        if np.is_traced((amount, vol_slurry)):
            return {key: self._per_volume(amount, vol_slurry) * density}

        if vol_slurry <= 0:
            return {}

        return {key: (amount / vol_slurry) * density}

    @staticmethod
    def _per_volume(amount, volume):
        """amount / volume where anything is held, else 0, for a trace.

        The flow terms below return no rate at all when the amount or the
        volume is zero. A trace cannot ask, so it computes the rate with a
        safe divisor and multiplies the empty case down to zero.
        """
        holding = volume > 0

        return np.where(holding, amount / np.where(holding, volume, 1.0), 0.0)

    def get_outlet_contributions(self, stream_phase, vessel_phase, amount,
                                 completed_state):

        vol = getattr(vessel_phase, 'vol', 0.0)

        if np.is_traced((amount, vol)):
            key = self._flow_state_key()
            state = completed_state.get(key)

            if state is None:
                state = getattr(self, self.flow_state_name)

            return {key: self._per_volume(amount, 0.0 if vol is None else vol)
                    * np.asarray(state, dtype=float)}

        if not amount:
            return {}

        vol = vol or 0.0

        if vol <= 0:
            # Nothing held, so nothing to withdraw. The population is zero
            # here in any case, so the term would vanish anyway; the guard
            # is only to keep amount/vol finite.
            return {}

        key = self._flow_state_key()
        state = completed_state.get(key)

        if state is None:
            state = getattr(self, self.flow_state_name)

        return {key: (amount / vol) * np.asarray(state, dtype=float)}

    def get_dilution_contributions(self, net_vol_flow, completed_state):
        """Dilution of the population as the slurry volume changes.

        The state n is per m3 of slurry, so from d(n V)/dt = Q_in n_in - Q_out n

            dn/dt = (Q_in n_in - Q_out n) / V  -  n (dV/dt) / V

        The inlet and outlet hooks give the first term. This one is the
        second, with dV/dt = Q_in - Q_out (``net_vol_flow``). It vanishes in
        an MSMPR whose volume is held, but in a semibatch crystallizer fed
        with liquid it is what keeps the growing volume from multiplying the
        crystal mass.
        """
        if not np.is_traced(net_vol_flow) and not net_vol_flow:
            return {}

        key = self._flow_state_key()
        state = completed_state.get(key)

        if state is None:
            state = getattr(self, self.flow_state_name)

        state = np.asarray(state, dtype=float)
        vol_slurry = self._slurry_volume(self.true_state(state))

        if np.is_traced((net_vol_flow, vol_slurry)):
            return {key: -self._per_volume(net_vol_flow, vol_slurry) * state}

        if vol_slurry <= 0:
            return {}

        return {key: -(net_vol_flow / vol_slurry) * state}


def van_leer_limiter(theta):
    """phi(theta) = (theta + |theta|) / (1 + |theta|).

    Second order and TVD, but |theta| puts a kink at theta = 0: the limited
    slope's derivative jumps where neighbouring differences change sign (the
    peak of a distribution, the edge of its support).
    """
    return (np.abs(theta) + theta) / (1 + np.abs(theta))


def van_albada_limiter(theta):
    """phi(theta) = (theta**2 + theta) / (theta**2 + 1).

    Second order like van Leer, and smooth everywhere: the limited slope
    phi(a/b) * b = (a**2 b + a b**2) / (a**2 + b**2) is a rational function
    of the neighbouring differences a, b, with no kink at a sign change. That
    is what derivative-based estimation wants from the right-hand side. The
    price is that it is not strictly TVD: for -1 < theta < 0 phi dips to
    -0.21, where van Leer is zero, so a small overshoot is possible at an
    extremum.
    """
    return (theta ** 2 + theta) / (theta ** 2 + 1)


FLUX_LIMITERS = {
    "van_leer": van_leer_limiter,
    "van_albada": van_albada_limiter,
}


class OneDFVMMechanism(PopulationBalanceMechanism):

    def __init__(
        self,
        owning_phase:BasePhase,
        target_components:str|list,
        solvent_name:str,
        x_grid,
        distrib_init:np.ndarray|None,
        kinetics=None,
        density=None,
        kv=1,
        distribution_state_name="distrib",
        scale=None,
        rad=None,
        distrib_basis="intensive",
        vol_slurry=None,
        basis_mass=None,
        limiter="van_leer",
    ):
        """
        Assumes an x_grid of constant dx

        Parameters
        ----------
        limiter : str, optional
            Flux limiter of the second-order upwind reconstruction, a key of
            FLUX_LIMITERS: ``'van_leer'`` (default) or ``'van_albada'``, the
            smooth one (see van_albada_limiter). Fixed at construction: a
            compiled model freezes it, and at an empty start state the two
            give the same right-hand side, so a change after compiling would
            go unnoticed.
        distrib_basis : str, optional
            What ``distrib_init`` is expressed in. ``'intensive'``, the
            default and this stack's storage convention, is a number density
            per m3 of slurry. The other three are input formats, converted on
            the way in so that what is stored is always intensive:

            - ``'extensive'``   absolute counts; needs ``vol_slurry``
            - ``'vol_perc'``    a volume-fraction shape, the legacy
              ``SolidPhase`` default; needs ``basis_mass`` and ``vol_slurry``
            - ``'mass_perc'``   a mass-fraction shape; same requirements

        vol_slurry : float, optional
            Slurry volume the extensive form refers to [m3]. Required for
            every basis but ``'intensive'``, because the conversion is a
            division by it and a solid on its own does not have one.
        basis_mass : float, optional
            Solid mass the fraction shape refers to [kg]. Required for
            ``'vol_perc'`` and ``'mass_perc'``, which carry only a shape.
        """
        super().__init__(
            owning_phase=owning_phase,
            target_components=target_components,
            solvent_name=solvent_name,
            kinetics=kinetics,
            density=density,
            kv=kv)

        self.x_grid = np.asarray(x_grid)

        if len(self.x_grid) < 2:
            raise ValueError(
                'A size grid needs at least two nodes to have a cell width.')

        if np.any(np.diff(self.x_grid) <= 0):
            raise ValueError(
                'The size grid must be strictly increasing.')

        if limiter not in FLUX_LIMITERS:
            raise ValueError(
                f"limiter={limiter!r} is not one of {sorted(FLUX_LIMITERS)}.")
        self.limiter = limiter

        self.x_grid_sq = self.x_grid**2
        self.x_grid_cu = self.x_grid**3
        self.dx = self._cell_widths(self.x_grid)
        # Nucleus size for the nucleation MASS term. This must equal the size
        # at which the boundary condition injects nuclei, which for this
        # discretisation is the first grid node: the flux at the left face is
        # G*n(0) = B, so the population gains B*x_grid[0]**3 of volume per
        # second. Accounting for it as B*rad_zero**3 with rad_zero = 0 -- as
        # the original does -- means the crystals gain mass the liquid never
        # gave up. Measured on an empty vessel, the population balance grew
        # mu_3 at 8.5e-5 kg/s while the mass transfer reported 6e-18 kg/s.
        #
        # Overridable for a genuinely different nucleus size, but it should
        # track the grid unless you know why it should not.
        self.rad = self.x_grid[0] if rad is None else rad
        self.distribution_state_name = distribution_state_name
        assert len(distrib_init)==len(x_grid), "x_grid and distrib must be the same length"

        distrib_init = self._as_intensive_distribution(
            distrib_init, distrib_basis, vol_slurry, basis_mass)

        setattr(self,distribution_state_name,distrib_init)
        # self.x_grid is in microns and distrib is counts per micron-bin per m3
        self.solver_states = (
            StateVariable(
                name=distribution_state_name,
                phaseref=None,
                dim=len(self.x_grid),
                units="#/(micron m3 slurry)",
                state_type="diff",
                limit_negative_inventory=False
            ),
        )

        self._reject_scale(scale)
        self.scale = 1
        self._update_exposed_attributes()

        # x_grid and distrib are this mechanism's own names. The other three
        # are the names the crystal size distribution has always had on a
        # SolidPhase, and the cake correlations, SolidPhase.getPorosity and
        # Slurry.getSolidsConcentr are all written against them. Exposing
        # them lets a refactored solid answer the same questions a legacy one
        # does, so none of that shared physics has to be rewritten or
        # duplicated -- and so its numbers cannot drift from the original's.
        self.expose('x_grid', 'x_distrib', 'moments', 'getMoments', 'kv')

    # ---------------- crystal size distribution, under its legacy names ----

    @property
    def x_distrib(self):
        """The size grid, in microns, under the name a SolidPhase uses."""
        return self.x_grid

    @property
    def moments(self):
        """The first four moments of the current distribution, in m**n."""
        return self.compute_moments(self.distrib, self.x_grid)

    def getMoments(self, x_distrib=None, distrib=None, mom_num=None):
        """
        Moments in m**n, matching SolidPhase.getMoments term for term.

        The scaling is the same on both sides -- compute_moments applies
        MOMENT_UNIT_FACTOR**n with the factor at 1e-6, and the legacy method
        applies (1e-6)**mom_ind -- so a caller gets identical numbers from a
        refactored solid and a legacy one. The argument shape follows the
        legacy signature, including returning a bare float for a single
        requested moment.
        """

        if x_distrib is None:
            x_distrib = self.x_grid

        if distrib is None:
            distrib = self.distrib

        moments = self.compute_moments(np.asarray(distrib, dtype=float),
                                       np.asarray(x_distrib, dtype=float))

        if mom_num is None:
            return moments

        if isinstance(mom_num, (int, np.integer)):
            return float(moments[mom_num])

        return np.array([moments[n] for n in mom_num])

    # ---------------- basis of the supplied distribution ----------------

    def _as_intensive_distribution(self, distrib, basis, vol_slurry,
                                   basis_mass):
        """
        Whatever the caller supplied, as an intensive number density.

        The two fraction bases carry only a shape, so they are first given a
        magnitude the way the legacy SolidPhase does -- which produces an
        ABSOLUTE number density -- and then divided by the slurry volume like
        any other extensive input. Doing it in that order keeps one copy of
        each conversion rather than a combined formula per basis.
        """

        distrib = np.asarray(distrib, dtype=float)

        if basis == "intensive":
            return distrib

        known = ("intensive", "extensive", "vol_perc", "mass_perc")

        if basis not in known:
            raise ValueError(
                f"Unknown distrib_basis {basis!r}. Choose one of {known}.")

        if basis in ("vol_perc", "mass_perc"):

            if basis_mass is None:
                raise ValueError(
                    f"distrib_basis={basis!r} gives only the shape of the "
                    "distribution, so basis_mass is needed to give it a "
                    "magnitude.")

            # Normalized the way SolidPhase.getDistribution normalizes it.
            shape = distrib / distrib.sum()

            if basis == "vol_perc":

                if self.density is None:
                    raise ValueError(
                        "distrib_basis='vol_perc' converts through the solid "
                        "density, which this mechanism does not know. Pass "
                        "density, or attach the mechanism to a phase first.")

                distrib = volume_fraction_to_number(
                    self.x_grid, self.dx, shape, basis_mass, self.density,
                    kv=self.kv)
            else:
                distrib = mass_fraction_to_number(
                    self.x_grid, shape, basis_mass, kv=self.kv)

        return self.to_intensive(distrib, vol_slurry)


    # ---------------- legacy conversion ----------------

    @classmethod
    def claims_legacy_phase(cls, legacy_phase):
        """An old SolidPhase built from a distribution belongs here."""
        return getattr(legacy_phase, 'distrib', None) is not None \
            and getattr(legacy_phase, 'x_distrib', None) is not None

    def to_legacy_state(self):
        """The distribution, spelled the way an old SolidPhase holds it.

        x_grid and x_distrib are both in microns and distrib is a number
        density in both stacks, so the arrays transfer unchanged. Passing
        distrib_type='num' stops the old constructor from reinterpreting
        them as a volume-percent distribution and rescaling.
        """
        return {
            'x_distrib': np.asarray(self.x_grid),
            'distrib': np.asarray(getattr(self, self.distribution_state_name)),
            'kv': self.kv,
        }

    @classmethod
    def from_legacy_phase(cls, legacy_phase, owning_phase, **kwargs):
        """Rebuild the mechanism from an old SolidPhase's own attributes."""
        if not cls.claims_legacy_phase(legacy_phase):
            return None

        # legacy_phase.distrib is an ABSOLUTE count -- SolidPhase stores the
        # extensive distribution -- and distrib_init is read as a number
        # density per m3 of slurry. Handing one straight to the other, as
        # this did, inflated the population by the slurry volume: a factor of
        # several hundred on a typical charge. The slurry volume comes from
        # the caller because a solid on its own does not have one.
        vol_slurry = kwargs.pop('vol_slurry', None)

        if vol_slurry is None:
            raise ValueError(
                "Converting an old SolidPhase needs the slurry volume it "
                "belonged to: its distribution is an absolute count and this "
                "stack stores a number density per m3 of slurry. Pass "
                "vol_slurry. MixedPhase.from_legacy works it out from the "
                "phases it is given.")

        return cls(
            owning_phase=owning_phase,
            x_grid=np.asarray(legacy_phase.x_distrib),
            distrib_init=np.asarray(legacy_phase.distrib),
            distrib_basis='extensive',
            vol_slurry=vol_slurry,
            kv=getattr(legacy_phase, 'kv', 1),
            **kwargs,
        )

    def integrate_over_size(self, integrand, x_grid=None):
        """Rectangle rule, because this scheme stores CELL AVERAGES.

        ``dcsd_dt = -diff(flux)/dx`` advances cell averages, so the moment
        consistent with that conservation law is ``sum(n * x**k * dx)``, with
        dx the per-cell width -- a scalar on a uniform grid, an array on a
        geometric one, broadcasting either way. The
        trapezoid rule gives the first and last nodes half weight -- and the
        nucleation boundary condition injects every nucleus into exactly the
        first cell, so the population balance gained mu_3 at twice the rate
        the mass transfer removed solute from the liquid. Measured as a clean
        0.5000 ratio over 400 samples before this changed.

        The original carries the same mismatch: SolidPhase.getMoments is a
        trapezoid over the same cell averages.
        """
        return np.sum(integrand * self.dx)

    def compute_second_moment(
            self,
            distrib,
        ):
            """
            Compute second moment from a number density distribution.
            """
    
            return self.integrate_over_size(distrib * self.x_grid_sq)
    def compute_third_moment(
            self,
            distrib,
        ):
        """
        Compute third moment from a number density distribution.
        """
        return self.integrate_over_size(distrib * self.x_grid_cu)
    # compute_third_moment integrates over x_grid in microns, so its result is
    # in micron**3 and this converts it to m**3. The solid inventory is then
    # density * kv * mu_3, with no volume factor: the original crystallizer
    # scales nucleation by the slurry volume, which makes the distribution an
    # absolute count rather than a per-volume density
    # (Crystallizers.py: "mu_3 is total, not by volume").
    VOLUME_UNIT_FACTOR = 1e-18

    def get_mass(self):
        # mu_3 is now m3 of crystal per m3 of slurry, so the inventory needs
        # the slurry volume to become a mass.
        distribution = self.true_state(
            getattr(self,self.distribution_state_name))
        m3 = self.compute_third_moment(distribution)
        return (
            self.getDensity()
            * self.kv
            * m3
            * self.VOLUME_UNIT_FACTOR
            * self._slurry_volume(distribution)
        )
    def set_mass(self, mass):

        if mass is None:
            return

        vol_slurry = self._slurry_volume(
            self.true_state(getattr(self,self.distribution_state_name)))

        if np.is_traced((mass, vol_slurry,
                         getattr(self, self.distribution_state_name))):
            self._rescale_traced(mass, vol_slurry)
            return

        if vol_slurry <= 0:
            if mass == 0:
                return
            raise ValueError(
                'Cannot set a crystal mass without knowing the slurry volume; '
                'the distribution is a number density per m3 of slurry.')

        target_m3 = mass / (
            self.getDensity()
            * self.kv
            * self.VOLUME_UNIT_FACTOR
            * vol_slurry
        )

        self.set_third_moment(target_m3)

    def growth_size_factor(self, growth_params):
        """(1 + beta*x)**alpha of size-dependent growth, on the size grid.

        Evaluated from the kinetics' parameters as they are now. It used to
        be cached when the kinetics were attached, which froze alpha and
        beta: set_params (and so Parameters.assign) never reached them, a
        traced model held them as constants with zero sensitivity, and
        kinetics given to the constructor never set the cache at all.
        """
        return (1 + growth_params[4] * self.x_grid) ** growth_params[3]

    def set_third_moment(self, target_m3):
        """Rescale the distribution to a given third moment, keeping its shape.

        Parameters
        ----------
        target_m3 : float
            Third moment to reach [um**3/m3 slurry].

        Raises
        ------
        ValueError
            If a nonzero target is asked of a distribution whose third moment
            is not positive, so there is no shape to rescale.

        Notes
        -----
        A zero target empties the population. An integrator holds a
        just-nucleating population only to its absolute tolerance, so the
        distribution can carry small negative densities whose third moment is
        negative; asking such a population for no crystals must give none,
        not an error.
        """
        distribution = getattr(self,self.distribution_state_name)

        if target_m3 == 0:
            setattr(self, self.distribution_state_name,
                    np.zeros_like(distribution))  # [#/(um m3 slurry)]
            return

        # target_m3 is a TRUE moment, so compare against the true state; the
        # ratio is then applied to the conditioned state, which is what is
        # actually stored.
        current_m3 = self.compute_third_moment(self.true_state(distribution))
        if current_m3 <= 0:
            raise ValueError("Cannot scale a distribution with zero third moment.")

        factor = target_m3 / current_m3

        setattr(self,self.distribution_state_name,distribution * factor)

    def _rescale_traced(self, mass, vol_slurry):
        """set_mass for a traced mass, with where in place of each branch.

        A zero target empties the population, a positive one rescales it,
        and with no slurry volume a zero mass leaves it alone. Where the
        numeric path raises -- a mass without a volume, or a target for an
        empty population -- a trace cannot, so the population becomes NaN
        and the integrator stops on it.
        """
        name = self.distribution_state_name
        distribution = getattr(self, name)

        known = vol_slurry > 0
        target_m3 = mass / (
            self.getDensity()
            * self.kv
            * self.VOLUME_UNIT_FACTOR
            * np.where(known, vol_slurry, 1.0)
        )
        current_m3 = self.compute_third_moment(self.true_state(distribution))
        positive = current_m3 > 0

        factor = np.where(
            target_m3 == 0, 0.0,
            np.where(positive,
                     target_m3 / np.where(positive, current_m3, 1.0),
                     np.nan))
        factor = np.where(known, factor, np.where(mass == 0, 1.0, np.nan))

        setattr(self, name, distribution * factor)

    @staticmethod
    def _cell_widths(x_grid):
        """Width of the finite volume cell around each grid node.

        A uniform grid gives one scalar. A geometric grid -- which is the
        natural choice here, because it resolves the small sizes where
        nucleation happens -- gives an array, with cell faces at the
        geometric mean of adjacent nodes and the outer two extrapolated by
        the grid ratio.

        This mirrors Phases.getDistribution and the Slurry setup in
        MixedPhases, which is where the original computes exactly the same
        thing; the refactor had reduced it to ``x_grid[1] - x_grid[0]``, a
        single scalar applied to every cell. On geomspace(1, 1500, 35) that
        is 0.24 micron standing in for cells up to 323 micron wide -- a 1347x
        misweighting of the largest cells, in both the flux divergence and
        the moment integrals.
        """
        widths = np.diff(x_grid)

        if np.allclose(widths, widths[0], rtol=1e-8):
            return widths[0]

        ratio = x_grid[1] / x_grid[0]
        faces = np.zeros(len(x_grid) + 1)
        interior = np.sqrt(x_grid[1:] * x_grid[:-1])

        faces[0] = interior[0] / ratio
        faces[-1] = interior[-1] * ratio
        faces[1:-1] = interior

        return np.diff(faces)

    # ---------------- flow terms ----------------
    #
    # The inlet and outlet halves live on PopulationBalanceMechanism so
    # this and the moments form behave identically. Only the two pieces
    # that genuinely differ are here.

    def _validate_inlet_state(self, value):
        """A feed distribution must be on this vessel's own size grid."""
        if value.shape != self.x_grid.shape:
            raise ValueError(
                'An inlet crystal size distribution has %d bins but this '
                'vessel discretises size into %d. Feeding a distribution '
                'between different grids would need interpolation, which '
                'is not implemented -- give both units the same x_grid.'
                % (value.size, self.x_grid.size))

        return value

    def solid_volume_fraction(self, value):
        """Solid volume per m3 of slurry, from a number density."""
        return np.as_float(self.kv
                           * self.compute_third_moment(value)
                           * self.VOLUME_UNIT_FACTOR)
    def solve_population_balance(
        self,
        liquid:BasePhase,
        solid:BasePhase,
        completed_state:dict[StateKey],
        time:float,
        connection:PhaseConnection
    ) -> TransferResult:
        t0=perf_counter()
        # The distribution lives on the phase that owns this mechanism (the
        # solid), not on the connection's sink. For crystallization the two
        # coincide, but the dissolution connection runs solid -> liquid, so
        # keying off the sink looked for 'distrib' on the liquid and raised
        # KeyError. Any crystallizer given dissolution kinetics hit this.
        statekey =  StateKey(self.distribution_state_name,self.owning_phase_ref)
        csd = completed_state[statekey]
        self._timers['statekey_construct'] = self._timers.get('statekey_construct',0)+perf_counter()-t0
        t0 = perf_counter()
        # Physics reads the true population, never the conditioned state.
        csd_true = self.true_state(csd)
        moms = self.compute_moments(csd_true,self.x_grid)
        self._timers['pop_balance_compute_moments'] = self._timers.get('pop_balance_compute_moments',0)+perf_counter()-t0

        # mu_2 is in m**2, matching the original's getMoments basis. The
        # size-dependent branch below needs the micron-based integral instead.
        mu2 = moms[2] #total surface area
        t0 = perf_counter()
        conc, supersat, solubility = self.compute_supersaturation(liquid)
        self._timers['pop_balance_compute_supersat'] = self._timers.get('pop_balance_compute_supersat',0)+perf_counter()-t0

        t0 = perf_counter()
        nucl, growth, dissol = (
            self.mechanism_kinetics.get_kinetics(
                conc,
                liquid.temp,
                self.kv,
                moms,
            )
        )
        self._timers['pop_balance_compute_kinetics'] = self._timers.get('pop_balance_compute_kinetics',0)+perf_counter()-t0
        # nucl is #/(s m3 slurry) and the distribution is a number density on
        # that same basis, so there is NO volume factor here. Multiplying by
        # vol_slurry made the distribution an absolute count, which in turn
        # made the moments fed back into get_kinetics extensive -- so
        # secondary nucleation, which depends on magma density, scaled with
        # vessel size. The volume returns once, on the mass transfer below.
        vol_slurry = self.slurry_volume_from(liquid, csd_true)
        nucl_intensive = nucl
        nucl = nucl_intensive
        impurity_factor = self.mechanism_kinetics.alpha_fn(conc) #TODO check if con is the right units
        growth *= impurity_factor
        t0 = perf_counter()
        gparams = self.mechanism_kinetics.params["growth"]

        boundary = nucl / np.maximum(growth, eps)
        f_aug = np.concatenate((np.array([boundary, boundary]), csd,
                                np.array([csd[-1]])))
        self._timers['pop_balance_compute_boundary'] = self._timers.get('pop_balance_compute_boundary',0)+perf_counter()-t0
        t0 = perf_counter()
        # Flux source terms
        f_diff = np.diff(f_aug)
        # Upwind ratio, from the side the crystals grow from. A trace takes
        # both and lets where choose, since it cannot ask the sign.
        if np.is_traced(growth):
            theta = np.where(growth > 0,
                             f_diff[:-1] / (f_diff[1:] + eps * 10),
                             f_diff[1:] / (f_diff[:-1] + eps * 10))
        elif growth > 0:
            theta = (f_diff[:-1]/ (f_diff[1:] + eps * 10))
        else:
            theta = (f_diff[1:]/ (f_diff[:-1] + eps * 10))

        # Flux limiter (van Leer unless the mechanism was built otherwise)
        limiter = np.zeros_like(f_diff)
        limiter = np.set_at(limiter, slice(None, -1),
                            FLUX_LIMITERS[self.limiter](theta))
        self._timers['pop_balance_compute_limiter'] = self._timers.get('pop_balance_compute_limiter',0)+perf_counter()-t0

        # t0 = perf_counter()
        # Constant growth
        if len(gparams) == 3:

            growth_term = growth* (f_aug[1:-1]+ 0.5 * f_diff[1:] * limiter[:-1])
            dissol_term = dissol* (f_aug[2:]- 0.5 * f_diff[1:] * limiter[1:])
            growth_int = growth * self.compute_second_moment(csd_true)

        # Size-dependent growth
        else:

            t0 = perf_counter()
            growth_dep = growth * self.growth_size_factor(gparams)
            self._timers['pop_balance_compute_growth_dep'] = self._timers.get('pop_balance_compute_growth_dep',0)+perf_counter()-t0
            t0 = perf_counter()
            growth_pad = np.append(growth_dep,growth_dep[-1],)
            self._timers['pop_balance_compute_growth_dissol_pad'] = self._timers.get('pop_balance_compute_growth_dissol_pad',0)+perf_counter()-t0
            t0 = perf_counter()
            growth_term = growth_pad* (f_aug[1:-1]+ 0.5 * f_diff[1:] * limiter[:-1])
            dissol_term = dissol* (f_aug[2:]- 0.5 * f_diff[1:] * limiter[1:])
            self._timers['pop_balance_compute_growth_dissol_term'] = self._timers.get('pop_balance_compute_growth_dissol_term',0)+perf_counter()-t0
            t0 = perf_counter()
            growth_int = self.integrate_over_size(
                growth_dep * csd_true * self.x_grid_sq)
            self._timers['pop_balance_compute_growth_dissol_int'] = self._timers.get('pop_balance_compute_growth_dissol_int',0)+perf_counter()-t0

        t0 = perf_counter()
        # d(mu_3)/dt = 3*G*mu_2 + B*rad**3, so the 3 belongs to the growth
        # and dissolution terms ONLY -- it used to distribute over the
        # nucleation term as well. Every term is micron-based here, so one
        # VOLUME_UNIT_FACTOR converts the lot (the original applied 1e-6 to a
        # metre-based mu_2 and to a micron-based rad**3 together, leaving the
        # nucleation term wrong by 1e12).
        #
        # This is a rate per m3 of slurry; vol_slurry makes it extensive.
        dissol_int = dissol * self.compute_second_moment(csd_true)

        mass_transfer = (
            self.density * self.kv
            * (3 * (growth_int + dissol_int) + nucl_intensive * self.rad**3)
            * self.VOLUME_UNIT_FACTOR
            * vol_slurry
        )
        self._timers['pop_balance_compute_mass_transfer'] = self._timers.get('pop_balance_compute_mass_transfer',0)+perf_counter()-t0
        # self._timers['pop_balance_handle_growth'] = self._timers.get('pop_balance_handle_growth',0)+perf_counter()-t0
        t0 = perf_counter()
        flux = growth_term + dissol_term
        self._timers['pop_balance_flux_sum'] = self._timers.get('pop_balance_flux_sum',0)+perf_counter()-t0
        t0 = perf_counter()
        dcsd_dt = -np.diff(flux) / self.dx
        self._timers['pop_balance_compute_dcsd_dt'] = self._timers.get('pop_balance_compute_dcsd_dt',0)+perf_counter()-t0
        t0 = perf_counter()
        aux = {
            "supersaturation": supersat,
            "solubility": solubility,
            "moments": moms,
            "growth": growth,
            "dissolution": dissol,
            "nucleation": nucl,
            "flux": flux,
        }
        self._timers['pop_balance_compute_flux'] = self._timers.get('pop_balance_compute_flux',0)+perf_counter()-t0
        t0 = perf_counter()
        species_rates_out = self.compute_species_transfer(mass_transfer,liquid)
        self._timers['pop_balance_compute_species_transfer'] = self._timers.get('pop_balance_compute_species_transfer',0)+perf_counter()-t0
        t0 = perf_counter()
        state_rates = {StateKey(self.solver_states[0].name,self.owning_phase): dcsd_dt} #if phaseref is a phase instead of a PhaseRef, the vessel will determine the phaseref
        # Species leave or enter the LIQUID, which is the connection's source
        # for crystallization but its sink for dissolution. Keying off the
        # source unconditionally would return dissolved mass to the solid.
        state_rates.update({StateKey('mass_j',self._liquid_phaseref(connection)):species_rates_out})
        result = TransferResult(state_rates=state_rates,aux=aux,net_mass_rate=mass_transfer)
        self._timers['pop_balance_format'] = self._timers.get('pop_balance_format',0)+perf_counter()-t0
        return result

class MomentsPopulationBalance(PopulationBalanceMechanism):
    """Method of moments, ported from Crystallizers.py::method_of_moments.

    Tracks mu_0..mu_{n-1} rather than a resolved distribution. Moments are
    carried in micron**n as the original does; kinetics are handed them in
    m**n per m**3 of suspension, and the mass transfer carries the
    (1e-6)**3 conversion.

    Caveat inherited from the original: the moment form was only ever
    exercised on the volume basis in the batch crystallizer. It was still
    part-built for the reactive crystallizer, which was the first move to a
    mass basis, so the mass-basis path here is unproven.
    """

    def __init__(
        self,
        owning_phase,
        target_components,
        solvent_name,
        moments_init,
        kinetics=None,
        density=None,
        kv=1,
        rad=0.0,
        scale=None,
        moments_state_name='mu_n',
    ):
        super().__init__(
            owning_phase=owning_phase,
            target_components=target_components,
            solvent_name=solvent_name,
            kinetics=kinetics,
            density=density,
            kv=kv,
        )

        moments_init = np.atleast_1d(np.asarray(moments_init, dtype=float))

        self.moments_state_name = moments_state_name
        self.num_mom = len(moments_init)

        # rad is the nucleus size. It lived on the old unit operation as
        # rad_zero (default 0), never on the phase, so it must be supplied.
        self.rad = rad
        self._reject_scale(scale)
        self.scale = 1

        setattr(self, moments_state_name, moments_init)

        # PopulationBalanceMechanism registers a derived 'mu_n' output,
        # computed by integrating a resolved distribution. Here the
        # moments ARE the solver state, so that output both duplicates it
        # and collides with it by name -- the vessel then has one name
        # meaning two different things.
        self.output_states = [state for state in self.output_states
                              if state.name != moments_state_name]

        self.solver_states = (
            StateVariable(
                name=moments_state_name,
                dim=self.num_mom,
                index=list(range(self.num_mom)),
                units='micron**n',
                state_type='diff',
                limit_negative_inventory=False,
            ),
        )

        self._update_exposed_attributes()

    # ---------------- legacy conversion ----------------

    @classmethod
    def claims_legacy_phase(cls, legacy_phase):
        """An old SolidPhase built from moments rather than a distribution."""
        return (getattr(legacy_phase, 'moments', None) is not None
                and getattr(legacy_phase, 'distrib', None) is None)

    def to_legacy_state(self):
        return {
            'moments': np.asarray(getattr(self, self.moments_state_name)),
            'num_mom': self.num_mom,
            'kv': self.kv,
        }

    @classmethod
    def from_legacy_phase(cls, legacy_phase, owning_phase, **kwargs):
        if not cls.claims_legacy_phase(legacy_phase):
            return None

        # Same basis change as the resolved-distribution bridge above: the
        # legacy moments are absolute, this stack's are per m3 of slurry, and
        # every moment scales with the distribution so one division does all
        # four.
        vol_slurry = kwargs.pop('vol_slurry', None)

        if vol_slurry is None:
            raise ValueError(
                "Converting an old SolidPhase needs the slurry volume it "
                "belonged to: its moments are absolute and this stack stores "
                "them per m3 of slurry. Pass vol_slurry. "
                "MixedPhase.from_legacy works it out from the phases it is "
                "given.")

        return cls(
            owning_phase=owning_phase,
            moments_init=to_intensive(np.asarray(legacy_phase.moments),
                                      vol_slurry),
            kv=getattr(legacy_phase, 'kv', 1),
            **kwargs,
        )

    # ---------------- population balance ----------------

    def get_mass(self):
        """Solid mass implied by the third moment."""
        moments = np.asarray(getattr(self, self.moments_state_name))

        if len(moments) < 4:
            return 0.0

        return (self.density * self.kv * self.true_state(moments)[3] * 1e-18
                * self._slurry_volume(self.true_state(moments)))

    def set_mass(self, value):
        """Rescale the moments to a crystal mass, keeping their ratios.

        Parameters
        ----------
        value : float
            Crystal mass in the vessel [kg].

        Raises
        ------
        ValueError
            If fewer than four moments are carried, if the slurry volume is
            unknown for a nonzero mass, or if a nonzero mass is asked of
            moments whose third moment is not positive.

        Notes
        -----
        With a slurry volume known, a zero mass empties the population, for
        the reason given in ``OneDFVMMechanism.set_third_moment``. Without
        one, a zero mass leaves the moments unchanged.
        """
        moments = np.asarray(getattr(self, self.moments_state_name),
                             dtype=float)

        if len(moments) < 4:
            raise ValueError(
                'Setting mass needs at least four moments (mu_0..mu_3)')

        vol_slurry = self._slurry_volume(self.true_state(moments))

        if np.is_traced((value, vol_slurry, moments)):
            # The branches below as where, as in OneDFVMMechanism; NaN
            # stands in for each error a trace cannot raise.
            known = vol_slurry > 0
            target_mu3 = value / (self.density * self.kv * 1e-18
                                  * np.where(known, vol_slurry, 1.0))
            current = self.true_state(moments)[3]
            positive = current > 0
            factor = np.where(
                target_mu3 == 0, 0.0,
                np.where(positive,
                         target_mu3 / np.where(positive, current, 1.0),
                         np.nan))
            factor = np.where(known, factor,
                              np.where(value == 0, 1.0, np.nan))
            setattr(self, self.moments_state_name, moments * factor)
            return

        if vol_slurry <= 0:
            if value == 0:
                return
            raise ValueError(
                'Cannot set a crystal mass without knowing the slurry '
                'volume; the moments are per m3 of slurry.')

        target_mu3 = value / (self.density * self.kv * 1e-18 * vol_slurry)
        current = self.true_state(moments)[3]

        if target_mu3 == 0:
            setattr(self, self.moments_state_name,
                    np.zeros_like(moments))  # [um**k/m3 slurry]
            return

        if current <= 0:
            raise ValueError(
                'Cannot scale moments with a zero third moment.')

        setattr(self, self.moments_state_name,
                moments * (target_mu3 / current))  # ratio is scale-invariant

    # ---------------- flow terms ----------------
    #
    # Inherited from PopulationBalanceMechanism. Old
    # MSMPR.material_balances applies the identical expression to the
    # moment set that it applies to a resolved distribution:
    #
    #     input_distrib = u_inputs['Inlet']['mu_n'] * (1e6)**arange(n)
    #     flow_distrib  = tau_inv * (input_distrib - distrib)
    #
    # so only the volume fraction differs, mu_3 being available directly
    # rather than needing to be integrated.

    def solid_volume_fraction(self, value):
        """Solid volume per m3 of slurry, from the third moment."""
        moments = np.atleast_1d(np.asarray(value, dtype=float))

        if len(moments) < 4:
            return 0.0

        return np.as_float(self.kv * moments[3] * self.VOLUME_UNIT_FACTOR)

    def solve_population_balance(
        self,
        liquid,
        solid,
        completed_state,
        time,
        connection,
    ):

        statekey = StateKey(self.moments_state_name, self.owning_phase_ref)
        mu = np.asarray(completed_state[statekey], dtype=float)

        conc, supersat, solubility = self.compute_supersaturation(liquid)

        # Intensive basis: mu is already per m3 of slurry, so the kinetics
        # need only the micron -> metre conversion, with no volume division.
        # The volume returns on the mass transfer below.
        mu_true = self.true_state(mu)
        mu_susp = mu_true * (1e-6) ** np.arange(len(mu_true))
        vol_slurry = self.slurry_volume_from(liquid, mu_true)

        nucl, growth, dissol = self.mechanism_kinetics.get_kinetics(
            conc, liquid.temp, self.kv, mu_susp)

        nucl_intensive = nucl
        nucl = nucl_intensive
        growth = growth * self.mechanism_kinetics.alpha_fn(conc)

        ind_mom = np.arange(1, len(mu))

        dmu_zero_dt = np.atleast_1d(nucl)
        dmu_1on_dt = (ind_mom * (growth + dissol) * mu[:-1]
                      + nucl * self.rad ** ind_mom)

        dmu_dt = np.concatenate((dmu_zero_dt, dmu_1on_dt))

        # Per m3 of slurry, then made extensive. mu_2 is micron**2 and
        # growth micron/s, so (1e-6)**3 converts to m3/s. The true moments
        # and the unscaled nucleation rate, not the conditioned state.
        mass_transfer = np.as_float(np.atleast_1d(
            self.density * self.kv
            * (3 * (growth + dissol) * mu_true[2]
               + nucl_intensive * self.rad ** 3)
        )[0]) * (1e-6) ** 3 * vol_slurry

        species_rates_out = self.compute_species_transfer(mass_transfer,
                                                          liquid)

        aux = {
            'supersat': supersat,
            'solubility': solubility,
            'growth': growth,
            'dissolution': dissol,
            'nucleation': nucl,
        }

        state_rates = {
            StateKey(self.moments_state_name, self.owning_phase): dmu_dt,
            StateKey('mass_j', self._liquid_phaseref(connection)):
                species_rates_out,
        }

        return TransferResult(state_rates=state_rates, aux=aux,
                              net_mass_rate=mass_transfer)









def method_of_moments(self, mu, conc, temp, params, rho_cry, vol=1):
        kv = self.Solid_1.kv # shape factor

        # Kinetics
        if self.basis == 'mass_frac':
            rho_liq = self.Liquid_1.getDensity()
            comp_kin = conc / rho_liq
        else:
            comp_kin = conc

        # Kinetic terms
        mu_susp = mu*(1e-6)**np.arange(self.num_distr) / vol  # m**n/m**3_susp
        nucl, growth, dissol = self.CrystKinetics.get_kinetics(comp_kin, temp, kv,
                                                          mu_susp)

        growth = growth * self.CrystKinetics.alpha_fn(conc)

        ind_mom = np.arange(1, len(mu))

        # Model
        dmu_zero_dt = np.atleast_1d(nucl * vol)
        dmu_1on_dt = ind_mom * (growth + dissol) * mu[:-1] + \
            nucl * self.rad**ind_mom
        dmu_dt = np.concatenate((dmu_zero_dt, dmu_1on_dt))

        # Material balance in kg_API/s --> G in um, u_2 in um**2 (or m**2/m**3)
        mass_transf = np.atleast_1d(rho_cry * kv * (
            3*(growth + dissol)*mu[2] + nucl*self.rad**3)) * (1e-6)**3

        return dmu_dt, mass_transf

def fvm_method(self, csd, moms, conc, temp, params, rho_cry,
                   output='dstates', vol=1):

    mu_2 = moms[2]
    #assumes solid1 is target
    kv_cry = self.Solid_1.kv # volumetric shape factor

    # Kinetic terms
    if self.basis == 'mass_frac':
        rho_liq = self.Liquid_1.getDensity()
        comp_kin = conc / rho_liq
    else:
        comp_kin = conc

    nucl, growth, dissol = self.CrystKinetics.get_kinetics(comp_kin, temp,
                                                        kv_cry, moms)

    nucl = nucl * self.scale * vol 

    impurity_factor = self.CrystKinetics.alpha_fn(conc)
    growth = growth * impurity_factor  # um/s 
    gparams = self.CrystKinetics.params['growth']
    

    # dissol = dissol  # um/s
    boundary_cond = nucl / np.maximum(growth, eps) # num/um or num/um/m**3 initial
    f_aug = np.concatenate(([boundary_cond]*2, csd, [csd[-1]])) # TODO adjust for reaction or handled by concentration? 

    # Flux source terms
    f_diff = np.diff(f_aug)
    
    # f_diff[f_diff == 0] = eps  # avoid division by zero for theta

    if growth > 0:
        theta = f_diff[:-1] / (f_diff[1:] + eps*10)
        # theta = f_diff[:-1] / (f_diff[1:] + eps)
        # theta = f_diff[:-1] / f_diff[1:]
    else:
        theta = f_diff[1:] / (f_diff[:-1] + eps*10)
        # theta = f_diff[:-1] / (f_diff[1:] + eps)
        # theta = f_diff[:-1] / f_diff[1:]
    # Van-Leer limiter
    limiter = np.zeros_like(f_diff)
    limiter[:-1] = (np.abs(theta) + theta) / (1 + np.abs(theta))
    if len(gparams)==3:
    
        growth_term = growth * (f_aug[1:-1] + 0.5 * f_diff[1:] * limiter[:-1])
        dissol_term = dissol * (f_aug[2:] - 0.5 * f_diff[1:] * limiter[1:])
    else:
        growth_dependent = growth * (1 + self.x_grid * gparams[4])**gparams[3]
        dissol_dependent = dissol * (1 + self.x_grid * 0) # TODO add size-dependent dissol params
        growth_pad = np.append(growth_dependent,growth_dependent[-1])
        dissol_pad = np.append(dissol_dependent, dissol_dependent[-1])
        growth_term = growth_pad * (f_aug[1:-1] + 0.5 * f_diff[1:] * limiter[:-1])
        dissol_term = dissol_pad * (f_aug[2:] - 0.5 * f_diff[1:] * limiter[1:])
    flux = growth_term + dissol_term

        
    if output == 'flux':
        return flux  # TODO: isn't it necessary to divide by dx?
    elif output=='dstates':
        dcsd_dt = -np.diff(flux) / self.dx

        # Material bce in kg_API/s --> G in um, mu_2 in m**2 (or m**2/m**3)
        # AKA R_v (rho_c*kv*d_mu3_d_t)
        # Handle stoich in material balance
        if len(gparams)==3:
            mass_transfer = rho_cry * kv_cry * (
                3*(growth + dissol)*mu_2 + nucl*self.rad**3) * (1e-6)
        else:
            r_m = self.x_grid
            mass_transfer_growth = np.trapezoid(growth_dependent*csd*r_m**2,r_m)
            mass_transfer_dissol = np.trapezoid(dissol_dependent*csd*r_m**2,r_m)
            mass_transfer_nucl = nucl*self.rad**3
            mass_transfer = rho_cry*kv_cry*3*(mass_transfer_dissol+mass_transfer_growth+mass_transfer_nucl)*1e-18
        return dcsd_dt, np.array(mass_transfer)