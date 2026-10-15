# -*- coding: utf-8 -*-
"""
Solid-liquid separation on the MultiPhaseVessel architecture.

The legacy `SolidLiquidSep` module predates the vessel refactor: each class
there re-implements its own `Phases` setter, state declaration, solver call,
event handling and result assembly, and none of them share a base. This module
puts the ones that fit onto `MultiPhaseVessel` instead, so they inherit the
state collections, the phase/connection machinery, the controller protocol, the
event handling and every integrator backend.

Not all of them fit, and this module does not pretend otherwise:

- `Filter` is a 0-D batch unit with an explicit ODE and one terminal event. It
  is a vessel child, and its physics is unchanged from the original.
- `DeliquoringStep` is a vessel too, with its 1-D cake profile living in a
  mechanism the way the crystallizer's distribution does. It keeps the
  original's dimensionless states, because the governing equations are written
  against them, but it integrates in seconds rather than in a dimensionless
  time - the vessel has no way to be told otherwise, and the change of
  variable is one constant.
- `DisplacementWashing` has no state vector and no integration at all - it
  evaluates a closed-form `erfc` solution on a mesh. A vessel child cannot
  express that, so it will keep its own solve and merely adopt this stack's
  I/O contract. Not yet ported.

`Carousel` is not carried over: it was an 11-line stub whose only method tested
`__module__ == 'SolidLiqSep'`, a module name that has not existed for some time.

The cake correlations live in `PharmaPy.CakePhysics`, shared with the legacy
module and with `Drying_Model`.
"""

import numpy as np

from PharmaPy.CakePhysics import get_alpha, get_sat_inf, upwind_fvm
from PharmaPy.Commons import trapezoidal_rule
from PharmaPy.DataClasses import IntraPhaseProcess, OperatingKey
from PharmaPy.DataClasses import PhaseMapping, PhaseRef, StateEvent
from PharmaPy.DataClasses import StateKey, StateVariable, StreamConnection
from PharmaPy.DataClasses import TransferResult
from PharmaPy.Mechanisms import Mechanism, PopulationBalanceMechanism
from PharmaPy.MultiPhaseVessel import MultiPhaseVessel
from PharmaPy.ProcessControl_Refactored import Controller

eps = np.finfo(float).eps * 1.1


class _BaseSolidLiquidSep(MultiPhaseVessel):
    """
    Shared scaffolding for the cake-forming separations.

    What these units have in common is a fixed filtration area and a solid
    phase that is retained rather than consumed: no mechanism creates or
    destroys crystals here, so only the liquid carries a differential
    inventory.
    """

    # Forwarded wholesale rather than re-declared, for the reason given in
    # Reactors_Refactored: repeating the base signature means every parameter
    # later added to MultiPhaseVessel silently stops reaching this unit.
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def default_diff_states_from_phases(self):
        """
        Only the liquid holds a differential inventory.

        The base marks the basis state of *every* phase differential, which
        for a separation would give the solid a `mass_j` rate that nothing
        supplies: the cake neither grows nor dissolves, it is only drained.
        Narrowing to the first liquid is the same move the crystallizer makes
        for the same reason - there the solid's inventory is the population
        balance's `distrib` rather than a species mass.
        """

        if any(
            state.state_type == "diff"
            for phasestate in self.phase_states
            for state in self.phase_states[phasestate.phaseref].states.values()
        ):
            return

        for phasestatevar in self.phase_states:
            if (phasestatevar.state.name == self.basis
                    and phasestatevar.phaseref == PhaseRef("liquid", 0)):
                phasestatevar.state.update_variable("state_type", "diff")

    def _material_state_definition(self):
        """
        The phase inventory, opted out of the generic positivity limiter.

        These units stop on a terminal event - the filter at the point where
        the expressible liquid is gone - and that event *is* the guard against
        draining a phase past what it holds. The vessel's limiter is a second,
        cruder one: it throttles the outlet whenever an inventory would not
        survive `positivity_horizon` seconds, and with the default one-second
        horizon that bites long before the event does.

        Measured on a charge that filters in 4.25 s: the limiter stretched it
        to 6.07 s, a 43% error that did not shrink with the grid, while the
        constants behind it agreed to 6e-5. On a slower charge filtering over
        229 s the same default was invisible. So it is not a tolerance to
        tune - the original has no limiter at all, and reproducing it means
        opting out rather than picking a horizon.

        Set here rather than on the compiled states because
        `define_solver_states` deep-copies these, so a later mutation of
        `phase_states` never reaches the copy the limiter actually reads.
        """

        state = super()._material_state_definition()
        state.update_variable("limit_negative_inventory", False)

        return state

    def get_heat_transfer_area(self):
        """
        The filtration area, which is fixed by the housing.

        The base computes `4*vol/diam + area_base`, which is the wetted wall
        of a stirred tank and has nothing to do with a filter; it also raises
        when `diam` is zero, which it is here.
        """

        return self.area_filt


class FiltrationController(Controller):
    """
    Drives the filtrate draw from Darcy's law.

    The filtrate rate is not a free operating condition - it is set by the
    pressure drop against the resistance of a cake that thickens as filtrate
    leaves. Expressing it as an operating condition rather than as an
    algebraic state is what keeps the unit an ordinary ODE, so it runs on
    every backend; a `state_type='alg'` formulation would confine it to the
    two backends that solve DAEs.
    """

    def actuate(self, time, completed_state, unit,
                resolved_inlets=None, resolved_outlets=None):

        # Outlet conditions are set on the second pass, once the inlet is
        # resolved - the protocol MultiPhaseVessel.get_operating_conditions
        # documents. A filter has no inlet, but the two passes run regardless.
        if resolved_inlets is None:
            return

        self.operating_conditions[
            OperatingKey(
                "vol_flow",
                connection=unit.filtrate_connection,
                port="outlet",
            )
        ] = unit.filtrate_volumetric_flow(completed_state)


class Filter(_BaseSolidLiquidSep):
    """
    Dead-end cake filtration.

    The model is classic cake-filtration theory and is unchanged from
    `SolidLiquidSep.Filter`:

        dV/dt = dP / [mu * (alpha*c*V/A**2 + Rm/A)]

    written on masses rather than volumes. What changed is everything around
    it: the cumulative filtrate is no longer a state of its own, because the
    vessel already integrates the liquid inventory and the filtrate is what
    has left it. The original encoded exactly that relation as
    `dmass_dt = [deriv, -deriv]`, i.e. two states constrained to sum to a
    constant; here the constraint is structural instead.

    Parameters
    ----------
    station_diam : float
        Diameter of the filter cross section [m].
    deltaP : float, optional
        Pressure drop across cake and medium [Pa]. A constructor argument
        rather than a `solve_unit` keyword: the vessel's `solve_unit` forwards
        unknown keywords to the integrator, where a stray `deltaP` would land
        in the solver options.
    alpha : float or callable, optional
        Specific cake resistance [m/kg]. Computed from the crystal size
        distribution when omitted; a callable is evaluated at `deltaP`.
    resist_medium : float, optional
        Filter medium resistance [1/m].
    slurry_div : float, optional
        Divides the charge into this many equal filtration cycles. As in
        `SolidLiquidSep.Filter`, the simulated time is that of one cycle;
        the whole charge is still filtered (the cycles run side by side).
    """

    oper_mode = "Batch"

    # The cake leaves as the retained contents; the filtrate is the flowing
    # outlet. Connections unwraps a dict outlet through this name.
    default_output = "cake"

    filtrate_connection = 0

    def __init__(self, station_diam, deltaP=1e5, alpha=None,
                 resist_medium=1e9, slurry_div=1, **kwargs):

        # A filter exchanges no heat and the base energy balance would demand
        # a Cp for the whole cake; the original had no energy balance at all.
        kwargs.setdefault("isothermal", True)
        kwargs.setdefault("controller", FiltrationController())

        super().__init__(**kwargs)

        self.station_diam = station_diam
        self.area_filt = station_diam**2 * np.pi / 4  # [m**2]

        self.deltaP = deltaP
        self.alpha = alpha
        self.r_medium = resist_medium
        self.slurry_div = slurry_div

        self._filtration_ready = False

    # ------------------------------------------------------------------
    # Filtration constants
    # ------------------------------------------------------------------

    def compile_structure(self):

        super().compile_structure()

        # Derived from the charge, so they are fixed once the phases are
        # known. compile_structure runs at the top of every solve, which is
        # also when the original recomputed them.
        self.compute_filtration_constants()

    def compute_filtration_constants(self):
        """
        The constants the filtration rate is written against.

        Reproduces `SolidLiquidSep.Filter.solve_unit` term for term. The two
        slurry quantities the legacy `Slurry` supplied as methods -
        `getSolidsConcentr()` and `getFractions()[0]` - are inlined here from
        the third moment, which is how `MixedPhases.Slurry` computed them:
        the solid volume fraction is `mu_3 * kv`.
        """

        liquid = self.Phases.get_phase_from_ref(PhaseRef("liquid", 0))
        solid = self.Phases.get_phase_from_ref(PhaseRef("solid", 0))

        # The crystals are counted per m3 of slurry, so their population
        # balance needs this filter's liquid to turn that into an inventory.
        # A crystallizer wires it in its CrystKinetics setter, which a filter
        # never calls, so without this the cake's crystal mass read zero.
        pbm = solid.get_mechanism(PopulationBalanceMechanism)
        if pbm is not None:
            pbm.liquid_phase = liquid

        epsilon = solid.getPorosity(diam_filter=self.station_diam)
        dens_sol = solid.getDensity()

        if self.alpha is None:
            self.alpha = get_alpha(solid, sphericity=1, porosity=epsilon,
                                   rho_sol=dens_sol)
        elif callable(self.alpha):
            self.alpha = self.alpha(self.deltaP)

        vol_frac_solid = float(solid.moments[3] * solid.kv)
        solid_conc = max(0.0, vol_frac_solid * dens_sol)
        frac_liq = 1.0 - vol_frac_solid

        # The slurry volume comes from the liquid and the solid fraction, the
        # way PopulationBalanceMechanism.slurry_volume_from derives it. The
        # solid's own `vol` cannot be used: it is derived from the slurry
        # volume in turn, so it reads zero until the vessel has wired the
        # mechanism to a liquid - which has not happened yet at compile time.
        # Using it here made the slurry 2% too small, and that error carried
        # into c_solids, vol_filtrate and the terminal event.
        vol_slurry = liquid.vol / (1.0 - vol_frac_solid) / self.slurry_div

        vol_liq_cake = (vol_slurry * solid_conc / dens_sol
                        * epsilon / (1 - epsilon))
        vol_liq_slur = vol_slurry * frac_liq

        vol_filtrate = vol_liq_slur - vol_liq_cake

        self.c_solids = vol_slurry * solid_conc / vol_filtrate

        self.dens_liq = liquid.getDensity()
        self.visc_liq = liquid.getViscosityMix()

        # The run ends when this much liquid has been expressed from one
        # cycle: what is left is held in the cake pores and no pressure drop
        # will remove it.
        self.mass_crit = vol_filtrate * self.dens_liq

        # The vessel holds the WHOLE charge, so the filtrate is counted
        # against all of its liquid (filtrate_mass divides it back into one
        # cycle). Counting it against one cycle's liquid, as before, made the
        # filtrate negative for slurry_div > 1 and the filter never drained.
        self.mass_liquid_init = vol_liq_slur * self.dens_liq * self.slurry_div

        # Slurry and crystal volume of the whole charge, to keep the crystal
        # count when the liquid around the crystals drains (see
        # conserve_cake_crystals). Taken from the population balance itself,
        # whose quadrature the crystal mass uses, rather than from
        # solid.moments (a trapezoid rule that differs by a few % on coarse
        # grids).
        self._vol_slurry_init = self._vol_crystals = None
        if pbm is not None and pbm.flow_state_name is not None:
            state = pbm.true_state(
                np.asarray(getattr(pbm, pbm.flow_state_name), dtype=float))
            self._vol_slurry_init = pbm.slurry_volume_from(liquid, state)
            self._vol_crystals = (pbm.solid_volume_fraction(state)
                                  * self._vol_slurry_init)

        self._filtration_ready = True

    # ------------------------------------------------------------------
    # Rates
    # ------------------------------------------------------------------

    def filtrate_mass(self, completed_state):
        """Cumulative filtrate of one cycle, from the liquid the vessel no longer holds."""

        held = np.sum(
            np.asarray(completed_state[StateKey("mass_j", PhaseRef("liquid", 0))],
                       dtype=float)
        )

        return (self.mass_liquid_init - held) / self.slurry_div

    def filtrate_mass_flow(self, completed_state):
        """
        Darcy's law through cake plus medium, verbatim from the original.

        `SolidLiquidSep.Filter.material_balance`:
            cake_term = alpha * c_solids * mass_filtr / (area_filt*rho)**2
            filt_term = resist / (area_filt*rho)
            deriv     = dP / visc / (cake_term + filt_term)
        """

        mass_filtr = self.filtrate_mass(completed_state)

        cake_term = (self.alpha * self.c_solids * mass_filtr
                     / (self.area_filt * self.dens_liq)**2)
        filt_term = self.r_medium / (self.area_filt * self.dens_liq)

        return self.deltaP / self.visc_liq / (cake_term + filt_term)

    def filtrate_volumetric_flow(self, completed_state):
        """The controller sets a volumetric draw; the model gives a mass one.

        Darcy's law gives the rate of one cycle; the vessel drains all
        `slurry_div` cycles of its charge at once.
        """

        return self.slurry_div * self.filtrate_mass_flow(completed_state) / self.dens_liq

    # ------------------------------------------------------------------
    # Vessel hooks
    # ------------------------------------------------------------------

    def configure_default_connections(self):
        """
        One flowing outlet, carrying the liquid only.

        The cake does not flow: it is what remains when the run ends, and the
        `Outlet` property below hands it over as the default output. Mapping
        the solid here instead would drain crystals through the filter medium.
        """

        if len(self.outlet_connections) > 0:
            return

        stream = self.Phases.to_stream()

        self.outlet_connections = [
            StreamConnection(
                stream=stream,
                phase_mappings=[
                    PhaseMapping(
                        source_phaseref=PhaseRef("liquid", 0),
                        sink_phaseref=PhaseRef("liquid", 0),
                    )
                ],
            )
        ]

    def get_events(self):
        """
        Stop when the expressible liquid is gone.

        The original raised TerminateSimulation from a hand-rolled
        `__state_event`/`__handle_event` pair; this is the same surface
        written against the shared event contract, so the integrator
        root-finds it and every backend handles it the same way.
        """

        events = list(super().get_events())

        def filtrate_exhausted(time, completed_state, unit):
            return unit.mass_crit - unit.filtrate_mass(completed_state)

        events.append(
            StateEvent(
                name="filtrate_exhausted",
                function=filtrate_exhausted,
                direction=-1,
                terminal=True,
                source=self,
            )
        )

        return events

    # ------------------------------------------------------------------
    # Outlets
    # ------------------------------------------------------------------

    @property
    def Outlet(self):
        """
        Both outlets, named.

        The original returned only the cake and dropped the filtrate on the
        floor - the source carries a literal `# TODO: the other outlet` where
        this is built. A filter has two products, so both are offered here.
        `default_output` keeps the cake as what a flowsheet picks up when it
        does not ask, so existing connections are unaffected.
        """

        conditions = self.outlet_conditions

        filtrate = None

        if conditions is not None and conditions.streams:
            filtrate = conditions.streams[0].stream
        elif self.outlet_connections:
            filtrate = self.outlet_connections[0].stream

        return {"cake": self.Phases, "filtrate": filtrate}

    @Outlet.setter
    def Outlet(self, outlet):
        outlet = outlet if isinstance(outlet, (list, tuple)) else [outlet]
        self._create_default_connections("outlet_connections", outlet)

    # ------------------------------------------------------------------
    # Legacy result aliases
    # ------------------------------------------------------------------

    def retrieve_results(self, time, solver_states):

        result = super().retrieve_results(time, solver_states)

        # Every existing assertion against a Filter reads timeProf[-1]; the
        # flowsheet tests do nothing else with it.
        self.timeProf = np.asarray(time)
        self.massProf = np.asarray(solver_states)

        self.conserve_cake_crystals()

        return result

    def conserve_cake_crystals(self):
        """
        Keep the crystals of the cake when the liquid around them drains.

        The population is a number density per m3 of slurry and is not a
        solver state here (a filter neither creates nor destroys crystals),
        so as the filtrate leaves, the same density over the smaller slurry
        volume would describe fewer crystals. Rescaling it by
        V_slurry,initial / (V_liquid,final + V_crystals) keeps the count, and
        with it the crystal mass the cake hands on.
        """

        vol_crystals = getattr(self, "_vol_crystals", None)

        if not vol_crystals:
            return

        liquid = self.Phases.get_phase_from_ref(PhaseRef("liquid", 0))
        solid = self.Phases.get_phase_from_ref(PhaseRef("solid", 0))
        pbm = solid.get_mechanism(PopulationBalanceMechanism)

        if pbm is None or pbm.flow_state_name is None:
            return

        pbm.liquid_phase = liquid
        factor = self._vol_slurry_init / (float(liquid.vol) + vol_crystals)
        state = np.asarray(getattr(pbm, pbm.flow_state_name), dtype=float)
        setattr(pbm, pbm.flow_state_name, state * factor)

        # Only once per charge: a second call must not rescale again.
        self._vol_crystals = None


class DeliquoringMechanism(Mechanism):
    """
    Wakeman deliquoring of a filter cake, as a profile-owning mechanism.

    The cake is discretised into ``num_nodes`` axial cells and the mechanism
    carries two differential states over them, exactly as the population
    balance carries a crystal distribution: a reduced saturation and a
    reduced concentration per species. That is what puts this model on the
    vessel at all - a unit's own states are per phase, and these are per
    depth.

    Both states are the original's dimensionless ones. Reduced saturation is
    ``(S - s_inf)/(1 - s_inf)`` and reduced concentration is
    ``(C - C_mean)/(rho_j - C_mean)``, and they are kept that way rather than
    re-dimensionalised because the governing equations are written against
    them; re-deriving the model in SI would change the numbers, which is the
    one thing a refactor of this kind must not do. The dimensional profiles
    are published as output states instead.

    What does change is the independent variable. The original marches in a
    dimensionless time ``theta``; the vessel integrates in seconds and has no
    way to be told otherwise. Since ``theta = t * theta_conv`` with
    ``theta_conv`` a constant of the cake, the conversion is one multiply on
    the right-hand side.

    The constants come from the unit, which is the only thing that knows both
    phases and the applied pressure drop.
    """

    SATURATION_STATE = "sat_red"
    CONCENTRATION_STATE = "conc_star"

    # Wakeman's pore size distribution index and relative permeability
    # exponent, as in the original.
    PORE_INDEX = 5
    PERMEABILITY_EXPONENT = 3.4

    def __init__(self, owning_phase, num_nodes, num_species):

        super().__init__()

        self.owning_phase = owning_phase
        self.num_nodes = int(num_nodes)
        self.num_species = int(num_species)

        setattr(self, self.SATURATION_STATE, np.ones(self.num_nodes))
        setattr(self, self.CONCENTRATION_STATE,
                np.zeros(self.num_nodes * self.num_species))

        self.solver_states = (
            StateVariable(
                name=self.SATURATION_STATE,
                dim=self.num_nodes,
                units="",
                state_type="diff",
                # A saturation profile is not an inventory the vessel can
                # run out of, and the positivity limiter would read it as
                # one. The population balance opts out for the same reason.
                limit_negative_inventory=False,
            ),
            StateVariable(
                name=self.CONCENTRATION_STATE,
                dim=self.num_nodes * self.num_species,
                units="",
                state_type="diff",
                limit_negative_inventory=False,
            ),
        )

        self._update_exposed_attributes()

        # Filled in by the unit at compile time.
        self.p_gas = None
        self.p_thresh = None
        self.sat_inf = None
        self.delta_z = None
        self.theta_conv = None

    def get_overrides(self):
        """This mechanism takes over none of the phase's own attributes."""
        return {}

    # ------------------------------------------------------------------
    # Physics
    # ------------------------------------------------------------------

    def material_balance(self, sat_star, conc_star):
        """
        The original's `DeliquoringStep.material_balance`, unchanged.

        Returns d(states)/d(theta): still on the dimensionless time, so the
        caller applies `theta_conv`. Keeping the conversion outside is what
        lets this stay term for term the same expression as the original.
        """

        lambd = self.PORE_INDEX

        # Changing the negative element for numerical issues, as the original
        # does - the exponent below is negative, so a negative saturation
        # would raise it to a fractional power.
        sat_star = np.where(sat_star < 0, eps, sat_star)

        sat_aug = np.append(sat_star, sat_star[-1])
        p_liq = (self.p_gas
                 - self.p_thresh * sat_aug**(-1 / lambd)) / self.p_thresh

        dpliq_dz = np.diff(p_liq)

        k_rl = sat_star**self.PERMEABILITY_EXPONENT

        q_liq = -k_rl * dpliq_dz  # Non-dimensional liquid flux

        sinf = self.sat_inf
        sat_fun = (1 - sinf) / (sat_star * (1 - sinf) + sinf)
        advection_vel = q_liq * sat_fun

        conc_bound = conc_star[0]  # dC/dt|_{z=0} = 0

        flux_sat = upwind_fvm(q_liq, boundary_cond=0)
        flux_conc = upwind_fvm(conc_star, boundary_cond=conc_bound)

        numerical_fluxes = np.column_stack((flux_sat, flux_conc))

        dstates_dtheta = -np.diff(numerical_fluxes, axis=0).T / self.delta_z
        dstates_dtheta[1:] = dstates_dtheta[1:] * advection_vel

        return dstates_dtheta.T

    # ------------------------------------------------------------------
    # Vessel interface
    # ------------------------------------------------------------------

    def get_solver_state_rates(self, process, phase, time, completed_state):

        if self.theta_conv is None:
            raise RuntimeError(
                "The deliquoring constants have not been computed. They come "
                "from the unit, which does it in compile_structure."
            )

        saturation = np.asarray(
            completed_state[StateKey(self.SATURATION_STATE, process.phaseref)],
            dtype=float,
        )

        concentration = np.asarray(
            completed_state[StateKey(self.CONCENTRATION_STATE,
                                     process.phaseref)],
            dtype=float,
        ).reshape(self.num_nodes, self.num_species)

        rates = self.material_balance(saturation, concentration)

        # d/dtheta to d/dt. theta = t * theta_conv, and theta_conv is fixed by
        # the cake, so this is the whole of the change of variable.
        rates = rates * self.theta_conv

        state_rates = {
            StateKey(self.SATURATION_STATE, process.phaseref): rates[:, 0],
            StateKey(self.CONCENTRATION_STATE, process.phaseref):
                rates[:, 1:].reshape(-1),
        }

        return TransferResult(
            state_rates=state_rates,
            aux={"phase": phase, "process": process},
            # Nothing crosses a phase boundary here: the liquid redistributes
            # within the cake and leaves through the outlet, so the mechanism
            # itself creates and destroys no mass.
            net_mass_rate=0.0,
        )


class DeliquoringStep(_BaseSolidLiquidSep):
    """
    Gas-blow deliquoring of a filter cake.

    A pressure drop is applied across a cake that a filter has already
    formed, and gas displaces liquid from the pores until the saturation
    reaches its irreducible value. The model is Wakeman's, unchanged from
    `SolidLiquidSep.DeliquoringStep`; what changed is that the profile now
    lives in a mechanism and the unit integrates in seconds.

    The cake geometry is given rather than read off a `Cake` container: the
    refactored phase stack has no `Cake`, and the geometry is a property of
    the filtration that produced the cake rather than of the phases in it.

    Parameters
    ----------
    num_nodes : int
        Axial cells through the cake.
    cake_vol : float
        Volume of the cake, solids plus pore liquid [m**3].
    alpha : float
        Specific cake resistance [m/kg].
    deltaP : float, optional
        Pressure drop applied across the cake [Pa].
    diam_unit : float, optional
        Diameter of the cross section [m].
    resist_medium : float, optional
        Filter medium resistance [1/m].
    saturation_init : float, optional
        Pore saturation the cake starts at [-]. Defaults to a full cake.
    p_atm : float, optional
        Ambient pressure downstream of the cake [Pa].
    """

    oper_mode = "Batch"

    def __init__(self, num_nodes, cake_vol, alpha, deltaP=1e5,
                 diam_unit=0.01, resist_medium=1e9, saturation_init=1.0,
                 p_atm=101325, **kwargs):

        # The original has no energy balance, and a cake has no stirred-tank
        # heat transfer area for the base one to use.
        kwargs.setdefault("isothermal", True)

        super().__init__(**kwargs)

        self.num_nodes = int(num_nodes)
        self.cake_vol = cake_vol
        self.alpha = alpha
        self.deltaP = deltaP

        self.diam_unit = diam_unit
        self.area_cross = np.pi / 4 * diam_unit**2
        self.area_filt = self.area_cross

        self.resist_medium = resist_medium
        self.saturation_init = saturation_init
        self.p_atm = p_atm

        self.cake_height = cake_vol / self.area_cross

        # Dimensionless depth, cell centres and faces, as the original.
        z_faces = np.linspace(0, 1, self.num_nodes + 1)

        self.z_grid = z_faces
        self.z_centers = (z_faces[1:] + z_faces[:-1]) / 2
        self.delta_z = np.diff(z_faces)

        self._mechanism = None

    # ------------------------------------------------------------------
    # Wiring
    # ------------------------------------------------------------------

    def default_diff_states_from_phases(self):
        """
        No phase inventory is integrated here.

        The original evolves the profile and nothing else - it never carries
        a liquid mass as a state - so marking `mass_j` differential would
        give the solver a state no balance supplies a rate for. The
        mechanism's two profiles are the whole state vector.
        """

        return

    def _post_set_phases(self):

        super()._post_set_phases()

        liquid_ref = PhaseRef("liquid", 0)
        liquid = self.Phases.get_phase_from_ref(liquid_ref)

        mechanism = DeliquoringMechanism(
            owning_phase=liquid,
            num_nodes=self.num_nodes,
            num_species=liquid.num_species,
        )

        self._mechanism = mechanism

        self.intraphase_processes = [
            IntraPhaseProcess(phaseref=liquid_ref, mechanism=mechanism)
        ]

    def define_output_states(self, overwrite=False):
        """
        Drop the outputs that need a phase inventory this unit has no state for.

        `mole_conc` is computed from `mass_j`, and `default_diff_states_from_phases`
        above deliberately declares no `mass_j`: the original evolves the cake
        profile and never carries a liquid mass. Leaving the output declared
        makes the results replay raise a KeyError the moment a solve finishes.

        The profiles themselves are recovered through `saturation_profile`
        and `concentration_profile`, which undo the reduction the states are
        stored in.
        """

        super().define_output_states(overwrite)

        states = self.output_state_collection.states

        for key in [key for key in states if key.name == "mole_conc"]:
            states.pop(key)

    # One state per node per species makes this large and sparse, the same
    # shape of problem the crystallizer's distribution makes.
    linear_solver_hint = "krylov"

    # ------------------------------------------------------------------
    # Constants
    # ------------------------------------------------------------------

    def compile_structure(self):

        super().compile_structure()

        self.compute_deliquoring_constants()

    def compute_deliquoring_constants(self):
        """
        Everything `DeliquoringStep.solve_unit` computed before integrating.

        Every expression is the original's. The two that touch the crystal
        size distribution - the irreducible saturation and the threshold
        pressure - are both ratios in the distribution, so they are unchanged
        by this stack storing an intensive distribution where the original
        stored an extensive one.
        """

        liquid = self.Phases.get_phase_from_ref(PhaseRef("liquid", 0))
        solid = self.Phases.get_phase_from_ref(PhaseRef("solid", 0))

        csd = solid.distrib
        diam_i = solid.x_distrib

        # mu_0 by the trapezoid rule, not solid.moments[0].
        #
        # It is used below only as the denominator that turns
        # int(p_thresh * csd) dx into a distribution-weighted average, and the
        # numerator is a trapezoid. This stack's `moments` come from the
        # rectangle rule, because the FVM stores cell averages, so taking mu_0
        # from there would divide a trapezoid by a rectangle. On the 1/x**4
        # integrand here that mismatch is not a rounding difference: it moved
        # p_thresh by a factor of 1.7 and theta_conv with it.
        mom_zero = trapezoidal_rule(diam_i, csd)

        epsilon = solid.getPorosity()

        rho_liq = np.mean(liquid.getDensity())
        self.visc_liq = np.mean(liquid.getViscosity())
        surf_tens = np.mean(liquid.getSurfTension())

        sat_inf = get_sat_inf(diam_i, csd, self.deltaP, epsilon,
                              self.cake_height, mom_zero,
                              (surf_tens, rho_liq))

        # Threshold pressure, averaged over the distribution.
        p_thresh = 4.6 * (1 - epsilon) * surf_tens / epsilon / diam_i
        p_thresh = trapezoidal_rule(diam_i, p_thresh * csd) / mom_zero

        rho_s = solid.getDensity()
        k_perm = 1 / self.alpha / rho_s / (1 - epsilon)

        deltaP_media = (self.deltaP * self.resist_medium
                        / (self.alpha * rho_s * self.cake_height
                           * (1 - epsilon) + self.resist_medium))

        pgas_out = self.p_atm + self.deltaP - deltaP_media

        mechanism = self._mechanism

        mechanism.p_gas = np.linspace(pgas_out, self.p_atm,
                                      self.num_nodes + 1)
        mechanism.p_thresh = p_thresh
        mechanism.sat_inf = sat_inf
        mechanism.delta_z = self.delta_z

        # Seconds to the dimensionless time the balance is written in.
        mechanism.theta_conv = (k_perm * p_thresh / self.visc_liq
                                / self.cake_height**2 / epsilon
                                / (1 - sat_inf))

        self.sat_inf = sat_inf
        self.p_thresh = p_thresh
        self.porosity = epsilon
        self.theta_conv = mechanism.theta_conv

        self.set_initial_profiles(liquid)

    def set_initial_profiles(self, liquid):
        """
        The initial reduced saturation and concentration.

        The concentration is reduced against its own depth-average, so a
        cake that starts uniform starts at zero - which is the original's
        behaviour and is why deliquoring only redistributes species once the
        saturation front has moved.
        """

        mechanism = self._mechanism

        saturation = np.full(self.num_nodes, self.saturation_init)
        sat_red = (saturation - self.sat_inf) / (1 - self.sat_inf)

        self.rho_j = liquid.getDensityPure()[0]

        conc_init = np.tile(np.asarray(liquid.mass_conc, dtype=float),
                            (self.num_nodes, 1))

        z_dim = self.z_centers * self.cake_height

        conc_mean_init = np.zeros_like(conc_init)

        for species in range(conc_init.shape[1]):
            conc_mean_init[:, species] = (
                trapezoidal_rule(z_dim, conc_init[:, species])
                / self.cake_height
            )

        self.conc_mean_init = conc_mean_init

        conc_star = (conc_init - conc_mean_init) / (self.rho_j
                                                   - conc_mean_init)

        setattr(mechanism, mechanism.SATURATION_STATE, sat_red)
        setattr(mechanism, mechanism.CONCENTRATION_STATE,
                conc_star.reshape(-1))

    # ------------------------------------------------------------------
    # Dimensional profiles
    # ------------------------------------------------------------------

    def saturation_profile(self, sat_red):
        """Reduced saturation back to a saturation."""

        return (np.asarray(sat_red, dtype=float) * (1 - self.sat_inf)
                + self.sat_inf)

    def concentration_profile(self, conc_star):
        """Reduced concentration back to kg/m**3."""

        conc_star = np.asarray(conc_star, dtype=float).reshape(
            self.num_nodes, -1)

        return (conc_star * (self.rho_j - self.conc_mean_init)
                + self.conc_mean_init)
