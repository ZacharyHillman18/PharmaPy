from PharmaPy.MultiPhaseVessel import MultiPhaseVessel
from PharmaPy.Mechanisms import *
from PharmaPy.DataClasses import *
from PharmaPy.ProcessControl_Refactored import DefaultContinuousVesselVolume

class _BaseCrystallizer(MultiPhaseVessel):

    # The population is a number density per m3 of slurry, so a feed that
    # grows the volume (a semibatch crystallizer) must dilute it.
    dilutes_intensive_states = True

    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        
    @property
    def CrystKinetics(self):
        raise AttributeError(
            "CrystKinetics is a convenience initializer only. Modify self.phase_connections directly instead."
            "Use phase_connections[ind].kinetics if the kinetics are desired."
        )

    @CrystKinetics.setter
    def CrystKinetics(self, instance: pk.CrystKinetics|list):
        ''' CrystKinetics is only a convenience initializer ONLY that assumes transfer between liquid1 and solid1 for each Crystkinetic in the list
            and that those connections are always active. It also assumes that target_comp is the only species moving for that index
            THIS CANNOT BE SET BEFORE PHASES
        If any more complex behavior is needed, or if phases need to be set after this, the user should set phase_connections directly using a list of PhaseConnection objects'''
        assert self._Phases is not None, 'Phases must be set before using the crystkinetics convenienve API. Otherwise, you must set phase_connections directly'
        if not isinstance(instance,list) and not isinstance(instance,pk.CrystKinetics):
            raise TypeError("CrystKinetics must be set by either a CrystKinetics object or a list of CrystKinetics objects")
        
        if not isinstance(instance,list):
            instance = [instance]
        self._CrystKinetics = instance
        self._create_default_phase_connections()
        self.nomenclature(overwrite=True)
        self._post_CrystKinetics_setter()
    def _create_default_phase_connections(self):
        ''' Assumes everything occurs between the first liquid and the first solid phases
        Assumes that whatever the target index is for that Crystkinetic, the massfrac is 100% that compound and 0 everything else
        Only runs if phase_connections are not already set'''

        if len(self.phase_connections)>0:
            raise RuntimeError(
                "phase_connections already defined. "
                "Cannot use CrystKinetics convenience API."
            )
        connections = []
        
        for i,ck in enumerate(self._CrystKinetics):
            solidphase_ref = PhaseRef('solid',0)
            try:
                solidphase=self.Phases.get_phase_from_ref(solidphase_ref)
            except IndexError:
                raise IndexError("The solid phase cannot be found, did you initialize the vessel with a solid phase?")
            # Matched on the base class, not OneDFVMMechanism: a
            # crystallizer discretised by moments carries a
            # MomentsPopulationBalance, and hardcoding the FVM class
            # silently returned None, so the moments form could not be
            # constructed at all.
            pbm = solidphase.get_mechanism(PopulationBalanceMechanism)

            if pbm is None:
                raise AttributeError(
                    'The solid phase carries no population balance mechanism. '
                    'Attach a OneDFVMMechanism or a MomentsPopulationBalance '
                    'to it before setting CrystKinetics.')
            pbm.liquid_phase_ref = PhaseRef("liquid",0)
            # The population is a number density per m3 of SLURRY, so the
            # mechanism needs the liquid to convert it into an inventory.
            self._bind_population_balance_liquid()
            pbm.owning_phase_ref = solidphase_ref
            weights = pbm.fraction
            if pbm._mechanism_kinetics is None:
                try:
                    ck.target_idx = pbm.target_ind[0]
                    pbm.mechanism_kinetics=ck
                except AttributeError:
                    raise AttributeError("Your solid phase does not have kinetics in its mechanism. The solidphase you assign to the vessel must have a mechanism with kinetics or crystallization cannot occur")
            reversible = ReversibleTransferMechanism(source_mechanism=pbm)
            
            if ck.supports(['growth','nucl_prim','nucl_sec']):
                # liquid to solid because crystallization is valid
                connection = PhaseConnection(source_phaseref=PhaseRef("liquid",0),
                                             sink_phaseref=solidphase_ref,
                                             kinetics=ck,
                                             species_weights=weights,
                                             active_condition=lambda source,sink:True,
                                             mechanism=reversible.forward
                                             )
                connections.append(connection)
            if ck.supports('dissolution'):
                #solid to liquid because dissolution
                connection = PhaseConnection(source_phaseref=PhaseRef("solid",0),
                                             sink_phaseref=PhaseRef('liquid',0),
                                             kinetics=ck,
                                             species_weights=weights,
                                             active_condition=lambda source,sink:True,
                                             mechanism=reversible.reverse

                                             )
                connections.append(connection)
        self.phase_connections = connections

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
                if phasestatevar.state.name==self.basis and phasestatevar.phaseref == PhaseRef("liquid",0):
                    phasestatevar.state.update_variable('state_type','diff')
    def _post_CrystKinetics_setter(self):
        "Place holder in case future children need special behavior"
        pass
    # The distribution block makes the system large and sparse, so an
    # iterative linear solve beats forming the dense Jacobian. A request in
    # neutral terms rather than a poke at the solver keeps this working for
    # any backend; one that has no such choice ignores it.
    linear_solver_hint = "krylov"
    def _post_set_phases(self):
        super()._post_set_phases()
        # Re-setting Phases after CrystKinetics (which is what SimulationExec
        # does when it hands the upstream liquid to a batch or semibatch
        # crystallizer) replaces the liquid object, so the population
        # balance must follow it. Otherwise its slurry volume, crystal mass
        # and outlet split keep coming from the discarded placeholder.
        self._bind_population_balance_liquid()

    def _bind_population_balance_liquid(self):
        """Point the solid's population balance at the vessel's live liquid.

        Does nothing until CrystKinetics has given the mechanism its
        liquid_phase_ref, or when the vessel has no solid or liquid.
        """
        try:
            solid = self.Phases.get_phase_from_ref(PhaseRef('solid', 0))
        except (IndexError, AttributeError):
            return

        pbm = solid.get_mechanism(PopulationBalanceMechanism)
        liquid_ref = getattr(pbm, 'liquid_phase_ref', None)

        if liquid_ref is None:
            return

        try:
            pbm.liquid_phase = self.Phases.get_phase_from_ref(liquid_ref)
        except (IndexError, AttributeError):
            return




class BatchCrystallizer(_BaseCrystallizer):

    oper_mode = "Batch"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)



class SemiBatchCrystallizer(_BaseCrystallizer):

    oper_mode = "Semibatch"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)



class ContinuousCrystallizer(_BaseCrystallizer):

    oper_mode = "Continuous"

    def __init__(
        self,
        controller=None,
        **kwargs
    ):

        # Constructed per instance; a default argument would be shared by
        # every ContinuousCrystallizer in the session.
        if controller is None:
            controller = DefaultContinuousVesselVolume()

        super().__init__(
            controller=controller,
            **kwargs
        )


    def configure_default_connections(self):

        if len(self.outlet_connections)>0:
            return


        stream = self.Phases.to_stream()

        mappings=[]

        counts={}

        for phase in self.Phases:

            phase_type = phase.phase_family.lower()

            idx = counts.get(
                phase_type,
                0
            )

            counts[phase_type]=idx+1


            ref = PhaseRef(
                phase_type,
                idx
            )

            mappings.append(
                PhaseMapping(
                    source_phaseref=ref,
                    sink_phaseref=ref
                )
            )


        self.outlet_connections=[
            StreamConnection(
                stream=stream,
                phase_mappings=mappings
            )
        ]