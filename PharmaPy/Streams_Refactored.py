from PharmaPy.Phases_Refactored import BasePhase
from PharmaPy.Interpolation import local_newton_interpolation
from PharmaPy.Results import DynamicResult
from PharmaPy.DataClasses import StateVariable,StateKey
import copy

from scipy.interpolate import CubicSpline
from PharmaPy import xp as np
# ===========================
# Legacy Streams content
# ===========================


def Interpolation(t_data, y_data, time, newton=True, num_points=3):
    """Kept as a module-level name; the implementation now lives in
    PharmaPy.Interpolation so the local-stencil logic exists once.
    This file used to carry its own copy, complete with the
    off-by-one that collapsed the stencil to a single node at the end
    of the data."""
    return local_newton_interpolation(time, t_data, y_data,
                                      num_points=num_points)


class BatchToFlowConnector:
    def __init__(self, cycle_time, flow_mult=1):
        self.flow_mult = flow_mult
        self.cycle_time = cycle_time

        self._Phases = None
        self._Inlet = None
        self.oper_mode = 'Batch'

        self.is_continuous = True

    @property
    def Phases(self):
        return self._Phases

    @Phases.setter
    def Phases(self, phases):
        self.has_solids = False

        if isinstance(phases, (list, tuple)):
            self._Phases = phases
        elif 'LiquidPhase' in phases.__class__.__name__:
            self._Phases = [phases]
        elif 'Slurry' in phases.__class__.__name__:
            self._Phases = phases.Phases
            self.has_solids = True

        classify_phases(self)
        self.nomenclature()

    def nomenclature(self):
        comp = self.Liquid_1.name_species
        self.states_di = {
            'mass_frac': {'dim': len(comp), 'index': comp},
            'mass_flow': {'dim': 1, 'units': 'kg/s'},
            'temp': {'dim': 1, 'units': 'K'}}

        self.names_states_out = ('temp', 'pres', 'mass_frac', 'mass_flow')

    def flatten_states(self):
        pass

    def solve_unit(self):
        self.retrieve_results()

    def retrieve_results(self):

        fields = ('temp', 'pres', 'mass_frac', 'path_data')

        # if self.cycle_time is None:
        #     self.cycle_time = self.Phases[0].time_upstream

        if self.has_solids:
            pass  # TODO: add this if continuous downstream solid processing is made available
        else:
            kw_phase = {key: getattr(self.Liquid_1, key) for key in fields}

            kw_phase['path_thermo'] = kw_phase.pop('path_data')
            mass_flow = self.Liquid_1.mass / self.cycle_time * self.flow_mult
            outlet = LiquidStream(**kw_phase, mass_flow=mass_flow)

            kw_phase.pop('path_thermo')
            kw_phase['mass_flow'] = mass_flow

        # kw_phase['time'] = [self.cycle_time]
        kw_phase['time'] = None

        self.result = DynamicResult(self.states_di, **kw_phase)

        self.Outlet = outlet
        self.outputs = kw_phase

# ================================================================
# New Stream Composition
# ================================================================


class BaseStream(BasePhase):
    phase_class = None
    def __init__(
        self,
        path_thermo=None,
        temp=298.15,
        pressure=101325,
        mass_flow=None,
        mole_flow=None,
        vol_flow=None,
        mass_j_flow=None,
        mass=None,
        moles=None,
        vol=None,
        mass_j=None,
        mass_frac=None,
        mole_frac=None,
        mass_conc=None,
        mole_conc=None,
        name_solv=None,
        controls=None,
        args_control=None,
        num_interpolation_points=3,
        check_input=True,
        verbose=True,
        **kwargs,
    ):
        self.is_stream=True
        if mass_flow is not None:
            mass = mass_flow
        if mole_flow is not None:
            moles = mole_flow
        if vol_flow is not None:
            vol = vol_flow
        if mass_j_flow is not None:
            mass_j = mass_j_flow
        super().__init__(
            path_thermo=path_thermo,
            temp=temp,
            pressure=pressure,
            mass=mass,
            moles=moles,
            vol=vol,
            mass_j=mass_j,
            mass_frac=mass_frac,
            mole_frac=mole_frac,
            mass_conc=mass_conc,
            mole_conc=mole_conc,
            name_solv=name_solv,
            check_input=check_input,
            verbose=verbose,
            **kwargs,
        )
        self._DynamicInlet = None
        self.controllable = ('mass_flow', 'mole_flow', 'vol_flow', 'temp') #Deprecation Warning
        self.time_upstream = None
        self.bipartite = None
        ## Deprecated control logic for compatibility
        # Controls
        if controls is None:
            controls = {}
        else:
            if args_control is None:
                args_control = {key: () for key in controls.keys()}

            update_dict = {}
            for key, fun in controls.items():
                update_dict[key] = fun(0, *args_control[key])

            self.updatePhase(**update_dict)

        self.controls = controls
        self.args_control = args_control

        self.num_interpolation_points = num_interpolation_points
        if self.phase_family is None:
            raise TypeError("BaseStream cannot be instantiated directly")
        
        
    # ====================================================
    # Aliases for flows
    # =====================================================
    @property
    def default_quantity_name(self):
        return 'mass_flow'
    @property
    def mass_flow(self):
        return self.mass
    @mass_flow.setter
    def mass_flow(self,value):
        if value is not None:
            self.mass = value
    @property
    def mole_flow(self):
        return self.moles
    @mole_flow.setter
    def mole_flow(self,value):
        if value is not None:
            self.moles = value
    @property
    def vol_flow(self):
        return self.vol
    @vol_flow.setter
    def vol_flow(self,value):
        if value is not None:
            self.vol = value
    @property
    def mass_j_flow(self):
        return self.mass_j
    @mass_j_flow.setter
    def mass_j_flow(self,value):
        if value is not None:
            self.mass_j = value


    # ========================================
    # Change key names for applicable methods
    # ==========================================
    
    def updatePhase(self,mass_flow=None,mole_flow=None,vol_flow=None,mass_j_flow=None, mole_conc=None, mass_conc=None, mass_frac=None, mole_frac=None, mass_j=None, mass=None, vol=None, moles=None, temp=None, pres=None, **kwargs):
        # 'is not None', not truthiness: these may be arrays, whose truth
        # value is ambiguous, and a legitimate zero flow would also read as
        # unspecified.
        if any(flow is not None and amount is not None
               for flow, amount in ((mass_flow, mass), (mole_flow, moles),
                                    (vol_flow, vol), (mass_j_flow, mass_j))):
            raise RuntimeError("Only a quantity or flow can be specified, not both")
        mass = mass_flow if mass_flow is not None else mass
        moles = mole_flow if mole_flow is not None else moles
        vol = vol_flow if vol_flow is not None else vol
        mass_j = mass_j_flow if mass_j_flow is not None else mass_j
        return super().updatePhase(
            mole_conc=mole_conc,
            mass_conc=mass_conc,
            mass_frac=mass_frac,
            mole_frac=mole_frac,
            mass_j=mass_j,
            mass=mass,
            vol=vol,
            moles=moles,
            temp=temp,
            pres=pres,
            **kwargs,
        )
    @property
    def state_dict(self):
        state_dict = super().state_dict
        state_dict['mass_flow'] = state_dict['mass']
        state_dict['vol_flow'] = state_dict['vol']
        state_dict['mole_flow'] = state_dict['moles']
        state_dict['mass_j_flow'] = state_dict['mass_j']
        return state_dict


    # ========================================
    # Input logic
    # ==========================================
    @property
    def DynamicInlet(self):
        return self._DynamicInlet

    @DynamicInlet.setter
    def DynamicInlet(self, dynamic_object):
        dynamic_object.controllable = self.controllable
        dynamic_object.parent_instance = self

        self._DynamicInlet = dynamic_object

    def InterpolateInputs(self, time):
        if isinstance(time, (float, int)):
            # Assume steady state for extrapolation
            time = min(time, self.time_upstream[-1])

            y_interpol = Interpolation(self.time_upstream, self.y_inlet,
                                        time,
                                        num_points=self.num_interpolation_points)
        else:
            interpol = CubicSpline(self.time_upstream, self.y_inlet)
            flags_interpol = time > self.time_upstream[-1]

            if any(flags_interpol):
                time_interpol = time[~flags_interpol]
                y_interp = interpol(time_interpol)

                y_extrapol = np.tile(y_interp[-1],
                                        (sum(flags_interpol), 1))
                y_interpol = np.vstack((y_interp, y_extrapol))
            else:
                y_interpol = interpol(time)

        return y_interpol
    def evaluate_inputs(self, time):
        if self.DynamicInlet is None:
            inputs = {}
            for attr in self.controllable:
                inputs[attr] = getattr(self, attr)

        else:
            inputs = self.DynamicInlet.evaluate_inputs(time)

        return inputs
    def add_stream_state_variables(self, collection):
        collection.add(
            StateVariable(
                name="vol_flow",
                dim=1,
                units="m3/s",
                stream="inlet"
            )
        )

        collection.add(
            StateVariable(
                name="temp",
                dim=1,
                units="K",
                stream="inlet"
            )
        )

        collection.add(
            StateVariable(
                name="mole_conc",
                dim=len(self.name_species),
                units="kmol/m3",
                stream="inlet"
            )
        )
    # A stream's pre-refactor counterpart takes flows rather than
    # amounts, so the conversion differs from a phase's only in which
    # keyword the quantity arrives under.
    LEGACY_QUANTITY_NAME = 'mass_flow'

    def to_legacy(self):
        """An equivalent stream from the pre-refactor PharmaPy.Streams."""
        import PharmaPy.Streams as legacy

        if self.LEGACY_CLASS_NAME is None:
            raise TypeError(
                f'{type(self).__name__} has no pre-refactor equivalent')

        legacy_cls = getattr(legacy, self.LEGACY_CLASS_NAME)

        kwargs = {
            'path_thermo': self.path_thermo,
            'temp': self.temp,
            'mass_frac': np.asarray(self.mass_frac).copy(),
            self.LEGACY_QUANTITY_NAME: float(self.mass_flow)
            if self.mass_flow is not None else 0,
        }

        if self.LEGACY_ACCEPTS_NAME_SOLV and self.name_solv is not None:
            kwargs['name_solv'] = self.name_solv

        mechanism_state = self._legacy_mechanism_state()
        kwargs.update(mechanism_state)

        return legacy_cls(**kwargs)

    def to_phase(self):
        phase = copy.copy(self)
        phase.__class__ = self.phase_class

        # composition_names and amount_names are CLASS attributes on
        # BasePhase, so mutating them in place rewrote them for every phase
        # in the process. Rebind to per-instance copies instead.
        phase.composition_names = set(self.composition_names) - {'mass_j_flow'}
        phase.amount_names = set(self.amount_names) - {
            'mass_flow', 'mass_j_flow', 'vol_flow', 'mole_flow'}

        phase.is_stream = False
        return phase


class LiquidStream(BaseStream):
    LEGACY_CLASS_NAME = 'LiquidStream'
    LEGACY_ACCEPTS_NAME_SOLV = True
    def __init__(
        self,
        path_thermo=None,
        temp=298.15,
        pressure=101325,
        mass_flow=None,
        mole_flow=None,
        vol_flow=None,
        mass_j_flow=None,
        mass_frac=None,
        mole_frac=None,
        mass_conc=None,
        mole_conc=None,
        name_solv=None,
        controls=None,
        num_interpolation_points=3,
        check_input=True,
        verbose=True,
        **kwargs,
    ):
        self.phase_family='liquid'
        from PharmaPy.Phases_Refactored import LiquidPhase
        self.phase_class = LiquidPhase
        super().__init__(
            path_thermo=path_thermo,
            temp=temp,
            pressure=pressure,
            mass_flow=mass_flow,
            mole_flow=mole_flow,
            vol_flow=vol_flow,
            mass_j_flow=mass_j_flow,
            mass_frac=mass_frac,
            mole_frac=mole_frac,
            mass_conc=mass_conc,
            mole_conc=mole_conc,
            name_solv=name_solv,
            controls=controls,
            num_interpolation_points=num_interpolation_points,
            check_input=check_input,
            verbose=verbose,
            **kwargs,
        )
        self.cp_liq = np.atleast_2d(self.cp_liq)
        self.p_vap = np.atleast_2d(self.p_vap)
class VaporStream(BaseStream):
    LEGACY_CLASS_NAME = 'VaporStream'
    phase_family = 'vapor'
    from PharmaPy.Phases_Refactored import VaporPhase
    phase_class = VaporPhase

class SolidStream(BaseStream):
    LEGACY_CLASS_NAME = 'SolidStream'
    phase_family='solid'
    from PharmaPy.Phases_Refactored import SolidPhase
    phase_class = SolidPhase

    #============================================================
    # Compatibility API: Deprecation Warning
    # ===========================================================
    def getPorosity(self, distrib=None, diam_filter=1, AR=None,
                    sphericity=None):

        if distrib is None:
            distrib = self.distrib
            mom_zero = self.moments[0]
            mom_one = self.moments[1]
        else:
            mom_zero, mom_one = self.getMoments(mom_num=(0, 1))

        # mom_one *= 1e-6  # m
        x_dist = self.x_distrib * 1e-6  # m

        if AR is None:
            AR = 2

        if sphericity is None:
            sphericity = 0.7

        # Yu, Zou et al (1996) and Yu,Zou, Stnadish (1996) model
        kv = 0.524  # Volumetric shape coefficient
        ks = 3.142  # Surface shape coefficient

        del_x_dist = np.diff(x_dist)
        node_x_dist = (x_dist[:-1] + x_dist[1:]) / 2
        node_CSD = (distrib[:-1] + distrib[1:]) / 2

        # Volume of crystals in each bin
        vol_cry = node_CSD * del_x_dist * (kv * node_x_dist**3)
        frac_vol_cry = vol_cry / (np.sum(vol_cry) + eps)

        vol_particle = kv * node_x_dist**3
        d_part_sphere = (6 * vol_particle / np.pi)**(1/3)
        d_part_equiv_pack = d_part_sphere / (sphericity**2.785 *
                                                np.exp(2.946 * (1 - sphericity)))

        # Initial porosity
        D_mean = mom_one/(mom_zero + eps)
        E_0_Jeschar = 0.375 + 0.34 * D_mean/diam_filter  # average porosity of packing of uniform sized spheres [-]

        initial_porosity = E_0_Jeschar

        V = 1/(1 - initial_porosity) * np.ones_like(node_x_dist)  # Specific Volume for initial porosity

        # Evaluate specific volume using modified linear packing model
        num_x = len(node_x_dist)
        V_T_node = np.zeros(num_x)

        for i in range(num_x):

            r = d_part_equiv_pack[:i] / d_part_equiv_pack[i]
            g_r = (1 - r)**2 + 0.4*r*(1 - r)**3.7
            V_large_j = V[:i] - (V[:i] - 1) * g_r - V[i]
            sum_V_large_term = sum(V_large_j * frac_vol_cry[:i])

            r_inv = r = d_part_equiv_pack[i] / d_part_equiv_pack[i + 1:]
            f_r = (1 - r_inv)**3.3 + 2.8*r_inv*(1 - r_inv)**2.7
            V_small_j = V[i + 1:] * (1 - f_r) - V[i]
            sum_V_small_term = sum(V_small_j * frac_vol_cry[i + 1:])

            V_T_node[i] = V[i] + sum_V_large_term + sum_V_small_term

        V_T = max(V_T_node)

        porosity = 1 - 1/V_T

        return porosity





#=======================================
# Legacy Functions
# =======================================



def classify_phases(instance, names=None):

    phases = instance.Phases

    if names is None:
        solid_count = 1
        liquid_count = 1
        vapor_count = 1

        for phase in phases:
            if 'Liquid' in phase.__class__.__name__:
                phase_name = 'Liquid_{}'.format(liquid_count)
                liquid_count += 1

            elif 'Solid' in phase.__class__.__name__:
                phase_name = 'Solid_{}'.format(solid_count)
                solid_count += 1

            elif 'Vapor' in phase.__class__.__name__:
                phase_name = 'Vapor_{}'.format(solid_count)
                vapor_count += 1

            setattr(phase, 'name', phase_name)
            setattr(instance, phase_name, phase)
    else:
        for phase, name in zip(phases, names):
            setattr(phase, 'name', phase_name)
            setattr(instance, name, phase)


def getPropsPhaseMix(phases, basis='mass'):
    # Empty containers
    props_matrix = np.zeros((len(phases), 3))
    vfrac_phases = []
    props_matrix = []

    for ind, phase in enumerate(phases):
        all_props = phase.getProps(basis=basis)
        props_matrix.append(all_props[:3])

        if phase.__class__.__name__ == 'LiquidPhase':
            ind_liq = ind
        if phase.__class__.__name__ == 'SolidPhase':
            mom_solid = all_props[-2]
            conv_exp = np.arange(len(mom_solid))
            # num/m**3, m/m**3, m**2/m**3, ...
            mom_meters = mom_solid * (1e-6)**conv_exp

            vfrac_solid = mom_meters[-1] * phase.kv
            vfrac_phases.append(vfrac_solid)

    props_matrix = np.vstack(props_matrix)

    # Volume fraction and mass fractions of phases
    vfrac_rest = 1 - sum(vfrac_phases)
    vfrac_phases.insert(ind_liq, vfrac_rest)

    # to avoid internal casting next line
    vfrac_phases = np.array(vfrac_phases)

    mass_phases = vfrac_phases * props_matrix[:, 1]
    mfrac_phases = mass_phases / mass_phases.sum()

    cp, rho, enthalpy = props_matrix.T

    return cp, rho, enthalpy, vfrac_phases, mfrac_phases

