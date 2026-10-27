from PharmaPy.DataClasses import (PhaseConnection,PhaseMapping,PhaseRef,PhaseStateCollection,PhaseStateVariable,
                                  IntraPhaseProcess,StateCollection,StateKey,StateVariable,StreamConnection,
                                  OperatingKey, StateEvent)
from typing import Any
import copy
from PharmaPy import xp as np
eps = np.finfo(float).eps


class Controller:

    # Solver states the controller integrates itself -- a PI loop's
    # integral, a filtered measurement -- declared as StateVariables, the way
    # a mechanism declares its own. The vessel adds them to its solver states
    # with no phase, as it does global_temp, and asks for their rates in
    # get_solver_state_rates. That is what gives a controller memory: the
    # value lives in the state vector, so it is integrated along the
    # accepted trajectory only, replayed exactly, carried through a
    # ConcurrentVessel with the member that owns the controller, and traced
    # like any other state. Memory kept in an attribute updated by actuate
    # would instead accumulate over every trial point the integrator
    # evaluates, and a trace would freeze it.
    #
    # A state declared with state_type='alg' is defined by a residual
    # (get_solver_state_residuals) rather than a rate, for a control law
    # that is implicit. That makes the vessel a DAE and needs a backend that
    # solves one.
    #
    # The initial value of each state is the controller attribute of the
    # same name, and the vessel keeps that attribute at the value last
    # evaluated, so a solve continued from the end of another starts where
    # it stopped.
    solver_states = ()

    def __init__(self):
        self.states = {}
        self.operating_conditions = {}

    def reset(self):
        self.states.clear()
        self.operating_conditions.clear()

    # ------------------------------------------------------------------
    # Solver states
    # ------------------------------------------------------------------

    @property
    def solver_state_keys(self):
        return tuple(StateKey(state.name) for state in self.solver_states)

    def owns_state(self, state_key):
        return state_key in self.solver_state_keys

    def add_solver_state_variables(self, collection, overwrite=False):
        """Register this controller's solver states with a vessel."""

        for state in self.solver_states:
            copied = copy.deepcopy(state)
            copied.phaseref = None
            collection.add(copied, overwrite)

    def initial_solver_state(self, state_key, unit=None):
        """Starting value of one of this controller's solver states."""

        return getattr(self, state_key.name, None)

    def update_solver_states(self, completed_state):
        """Hold each solver state's attribute at the value just evaluated."""

        for key in self.solver_state_keys:

            if key in completed_state:
                setattr(self, key.name, completed_state[key])

    def get_solver_state_rates(
        self,
        time,
        completed_state,
        unit,
        resolved_inlets=None,
        resolved_outlets=None,
        operating_conditions=None,
    ):
        """
        {StateKey: d(state)/dt} for every differential solver state.

        Called after the balances, with the streams and operating conditions
        of the same evaluation, so a rate can use what was actually applied
        (an outlet the limiter throttled, a set point held at its limit).
        """

        return {}

    def get_solver_state_residuals(
        self,
        time,
        completed_state,
        unit,
        resolved_inlets=None,
        resolved_outlets=None,
        operating_conditions=None,
    ):
        """{StateKey: residual} for every solver state declared 'alg'."""

        return {}

    def compute_states(
        self,
        time,
        completed_state,
        unit,
    ):

        self.states = {}

        self.update_state(
            time,
            completed_state,
            unit,
        )

        return self.states.copy()


    def observe(
        self,
        time,
        completed_state,
        unit,
        resolved_inlets=None,
        resolved_outlets=None,
    ):
        """
        Update controller measurements/internal state.

        This should not modify operating conditions.
        """

        pass


    def compute_operating_conditions(
        self,
        time,
        completed_state,
        unit,
        resolved_inlets=None,
        resolved_outlets=None,
    )->dict[OperatingKey,Any]:

        self.operating_conditions = {}

        self.actuate(
            time,
            completed_state,
            unit,
            resolved_inlets,
            resolved_outlets,
        )

        return self.operating_conditions.copy()


    def actuate(
        self,
        time,
        completed_state,
        unit,
        resolved_inlets=None,
        resolved_outlets=None,
    ):
        """
        Set manipulated variables through operating_conditions.
        """

        pass


    def update_state(
        self,
        time,
        completed_state,
        unit,
    ):
        pass
    def get_events(self,unit):
            return []
class TemperatureProfileMixin:
    """
    Drive the vessel temperature from a time profile.

    Owning ``global_temp`` replaces the vessel's energy balance: the vessel
    checks whether the key is present in ``states`` (see
    ``MultiPhaseVessel.has_energy_balance``) while it lays out its solver
    states, so the key has to be registered at construction, long before any
    ``update_state`` call.

    Mix this in ahead of the controller it extends so that its ``reset`` and
    ``update_state`` run first and still defer to the base controller.
    """

    temp_func = None
    temp_0 = None

    def set_temperature_profile(self, temp_func, temp_0=None):

        self.temp_func = temp_func

        # The seed value is only what the key holds until the first
        # update_state overwrites it; evaluating the profile keeps a reset
        # vessel consistent with the run that follows.
        self.temp_0 = temp_func(0.0) if temp_0 is None else temp_0

        self.states[StateKey("global_temp")] = self.temp_0

    def reset(self):

        super().reset()

        if self.temp_func is not None:
            self.states[StateKey("global_temp")] = self.temp_0

    def update_state(self, time, completed_state, unit):

        super().update_state(time, completed_state, unit)

        if self.temp_func is not None:
            self.states[StateKey("global_temp")] = np.of_time(self.temp_func,
                                                              time)


class SimpleTemperatureController(TemperatureProfileMixin, Controller):
    """Follow a temperature profile; leave all flows alone."""

    def __init__(self, temp_func, temp_0=None):
        super().__init__()
        self.set_temperature_profile(temp_func, temp_0)


class DefaultContinuousVesselVolume(Controller):
    """
    Hold vessel volume by trimming the outlet around the inlet flow.

    The outlet request is ``inlet + K * (V - V_target)``, so the volume
    error decays with time constant ``1 / K``. That time constant is the
    fastest mode in the model, and it is created entirely by this
    controller rather than by the process: pushing it far below the
    residence time buys no extra level control and costs the integrator
    proportionally many steps.

    ``tau`` therefore sets the gain from a settling time rather than
    exposing a bare gain. The default settles a level upset in a hundredth
    of a residence time, which is tight control and still ~1e4 times less
    stiff than a gain of 1e4 on a vessel of this size.
    """

    def __init__(
        self,
        target_volume=None,
        tau=None,
        K=None,
    ):

        super().__init__()

        self.target_volume = target_volume
        self.tau = tau
        self.K = K

        # Blend width, as a fraction of the inlet flow, over which the
        # no-backflow floor is applied. A hard max() puts a kink in the
        # right-hand side at the point the controller normally sits on.
        self.floor_width = 1e-3


    def observe(
        self,
        time,
        completed_state,
        unit,
        resolved_inlets=None,
        resolved_outlets=None,
    ):

        if self.target_volume is None:
            self.target_volume = unit.Phases.vol


    def get_gain(self, inlet_flow):
        """
        Proportional gain, in 1/s.

        An explicit K wins. Otherwise the gain is derived from the
        residence time, so the controller stays equally tight on vessels of
        any size instead of being ferociously stiff on small ones.
        """

        if self.K is not None:
            return self.K

        tau = self.tau

        if tau is None:

            # 'inlet_flow > 0 and target_volume', as where: an unset or
            # zero target falls back to one second.
            target = 0.0 if self.target_volume is None else self.target_volume
            flowing = (inlet_flow > 0) & (target != 0)
            residence_time = np.where(
                flowing, target / np.where(flowing, inlet_flow, 1.0), 1.0)

            tau = 0.01 * residence_time

        return 1.0 / tau

    @staticmethod
    def soft_floor(value, width):
        """
        C1-continuous stand-in for ``max(value, 0)``.

        Equals ``value`` above ``width`` and ``0`` below ``-width``, with a
        quadratic bridge whose slope matches at both ends. The controller
        sits near its floor whenever the vessel is at target, so a hard
        corner there is one the integrator keeps rediscovering.
        """

        # The three pieces of the C1 floor, selected with where so that a
        # trace sees all of them.
        safe_width = np.where(width > 0, width, 1.0)
        bridge = (value + safe_width) ** 2 / (4.0 * safe_width)
        smooth = np.where(value >= width, value,
                          np.where(value <= -width, 0.0, bridge))

        return np.where(width > 0, smooth, np.maximum(value, 0.0))


    def actuate(
        self,
        time,
        completed_state,
        unit,
        resolved_inlets=None,
        resolved_outlets=None,
    ):

        if resolved_inlets is None:
            return

        inlet_flow = self.total_inlet_flow(unit, resolved_inlets)

        volume_error = unit.Phases.vol - self.target_volume

        outlet_flow = inlet_flow + self.get_gain(inlet_flow) * volume_error

        self.operating_conditions[
            OperatingKey(
                "vol_flow",
                connection=0,
                port='outlet'
            )
        ] = self.soft_floor(
            outlet_flow,
            self.floor_width * np.maximum(inlet_flow, eps),
        )

    @staticmethod
    def total_inlet_flow(unit, resolved_inlets=None):
        """
        Volumetric feed into the vessel, summed over every inlet.

        Summed across all connections rather than reading the first one: a
        vessel may have several feeds, and a batch vessel has none at all, in
        which case indexing the first would raise.
        """

        if resolved_inlets is None:
            # Evaluated outside the resolve cycle (an event), so fall back to
            # the connections' own streams.
            return sum(
                phase.vol_flow
                for connection in unit.inlet_connections
                for phase in connection.stream
            )

        return sum(
            transfer.stream_phase.vol_flow
            for connection in resolved_inlets
            for transfer in connection
        )

    def requested_outlet_flow(self, unit, resolved_inlets=None,
                              completed_state=None):
        """
        The outlet the loop asks for before the no-backflow floor.

        Shared by actuate and the switching event so both describe the same
        surface. ``completed_state`` is for a subclass whose law reads its
        own solver states; this one does not.
        """

        inlet_flow = self.total_inlet_flow(unit, resolved_inlets)

        target = self.target_volume

        if target is None:
            target = unit.Phases.vol

        return inlet_flow + self.get_gain(inlet_flow) * (unit.Phases.vol - target)

    def get_events(self, unit):
        """
        The outlet switches between flow and no flow when the requested flow
        crosses zero, which is where soft_floor stops being the identity.
        Locating that crossing by root-finding is cheaper than discovering it
        through rejected steps.

        Skipped when there is nothing to control. On a vessel with no outlet
        the requested flow is identically zero, and a root function that is
        always zero is degenerate - Sundials warns about it and gains
        nothing.
        """

        if not unit.outlet_connections:
            return []

        def outlet_flow_sign(time, completed_state, unit):
            # The feed actually arriving, not the configured inlet stream:
            # a vessel fed from an upstream trajectory or a coupled unit
            # would otherwise place this surface using a flow it never sees.
            resolved_inlets = unit.resolve_current_inlets(
                time, completed_state)

            return float(self.requested_outlet_flow(unit, resolved_inlets,
                                                    completed_state))

        return [
            StateEvent(
                name="outlet_flow_sign",
                function=outlet_flow_sign,
                direction=0,
                terminal=False,
                source=self,
            )
        ]

class ContinuousVesselController(
    TemperatureProfileMixin,
    DefaultContinuousVesselVolume,
):
    """
    Hold vessel volume with the outlet while driving temperature on a profile.

    This is the pairing a continuous crystallizer normally wants: the outlet
    trims the level around the feed, and the jacket is assumed capable of
    tracking whatever temperature trajectory is asked for, so the energy
    balance is replaced by the profile rather than solved.

    Because ``global_temp`` becomes a controlled state, ``Utility`` is not
    consulted and the vessel carries one fewer differential state. Use
    DefaultContinuousVesselVolume on its own if you want the jacket duty to
    determine the temperature instead.

    Parameters
    ----------
    temp_func : callable
        ``temp_func(time) -> K``.
    temp_0 : float, optional
        Seed temperature. Defaults to ``temp_func(0)``.
    target_volume : float, optional
        Volume to hold. Defaults to the vessel's volume at the first
        evaluation.
    tau, K : float, optional
        Level-loop tuning, as in DefaultContinuousVesselVolume.
    """

    def __init__(
        self,
        temp_func,
        temp_0=None,
        target_volume=None,
        tau=None,
        K=None,
    ):

        super().__init__(
            target_volume=target_volume,
            tau=tau,
            K=K,
        )

        self.set_temperature_profile(temp_func, temp_0)


# ======================================================================
# PI(D) control: controllers whose memory is solver states
# ======================================================================

class PIDLoop:
    """
    The arithmetic of one PI(D) loop, its memory kept as solver states.

    ISA (standard) form, around a bias:

        u = bias + Kc * (e + I / Ti + Td * s * dy_f/dt)

        e = s * (y - y_set)

    with s = +1 for direct action (an outlet holding a level: more level,
    more outflow) and s = -1 for reverse action (a jacket heating a vessel:
    temperature below its set point, hotter jacket).

    I, the integral of e, and y_f, the measurement through a first-order
    filter of time constant Tf, are solver states named ``<name>_integral``
    and ``<name>_filtered``. The derivative acts on the filtered
    measurement: a set point step does not kick the output, and
    dy_f/dt = (y - y_f) / Tf needs no derivative of the right-hand side.

    Anti-windup is back-calculation. While the applied output differs from
    the law's -- held at a limit, or throttled by the vessel --

        dI/dt = e + Ti / (Kc * Tt) * (u_applied - u)

    pulls the integral back until the law asks for what is applied, with
    time constant Tt (default Ti).

    The loop stores no tuning. The controller that owns it passes the gain
    and times on every call, so they may be derived from the vessel: a
    residence time, a heat-transfer time constant.
    """

    def __init__(self, name, action="reverse", integral=True,
                 derivative=False, units=""):

        if action not in ("direct", "reverse"):
            raise ValueError("action must be 'direct' or 'reverse'.")

        self.name = name
        self.sign = 1.0 if action == "direct" else -1.0
        self.units = units

        self.integral_key = StateKey(f"{name}_integral") if integral else None
        self.filtered_key = (StateKey(f"{name}_filtered") if derivative
                             else None)

    @property
    def solver_states(self):

        states = []

        if self.integral_key is not None:
            states.append(StateVariable(
                name=self.integral_key.name, dim=1,
                units=f"({self.units}) s", state_type="diff",
                limit_negative_inventory=False))

        if self.filtered_key is not None:
            states.append(StateVariable(
                name=self.filtered_key.name, dim=1, units=self.units,
                state_type="diff", limit_negative_inventory=False))

        return tuple(states)

    def error(self, setpoint, measurement):
        return self.sign * (measurement - setpoint)

    def law(self, completed_state, setpoint, measurement, gain,
            integral_time=None, derivative_time=None, filter_time=None,
            bias=0.0):
        """The output the law asks for, before any limit."""

        action = self.error(setpoint, measurement)

        if self.integral_key is not None:
            action = action + completed_state[self.integral_key] / integral_time

        if self.filtered_key is not None:
            slope = ((measurement - completed_state[self.filtered_key])
                     / filter_time)
            action = action + derivative_time * self.sign * slope

        return bias + gain * action

    def rates(self, completed_state, setpoint, measurement, law, applied,
              gain, integral_time=None, filter_time=None, tracking_time=None):
        """{StateKey: rate} for this loop's solver states."""

        rates = {}

        if self.integral_key is not None:
            tracking = integral_time if tracking_time is None else tracking_time

            rates[self.integral_key] = (
                self.error(setpoint, measurement)
                + integral_time / (gain * tracking) * (applied - law))

        if self.filtered_key is not None:
            rates[self.filtered_key] = (
                (measurement - completed_state[self.filtered_key])
                / filter_time)

        return rates

    def initial_filtered(self, measurement):
        """The filter starts at the measurement, so no derivative kick."""
        return measurement


class PIDContinuousVesselVolume(DefaultContinuousVesselVolume):
    """
    Hold vessel volume with a PI(D) loop on the outlet.

        outlet = inlet + Kc * (e + I / Ti + Td * dV_f/dt),   e = V - V_target

    then the same C1 no-backflow floor as DefaultContinuousVesselVolume,
    whose proportional law this extends. The proportional law leaves an
    offset whenever the outflow has to differ from the inflow at steady
    state: a reacting liquid that densifies, a crystallizer whose slurry
    carries solid out. The volume then settles at (F_out - F_in) / Kc away
    from target. The integral removes the offset, so the volume is held at
    its target exactly.

    The integral (and, with derivative action, the filtered volume) is a
    solver state of the vessel; see Controller.solver_states.

    Parameters
    ----------
    target_volume, tau, K :
        As in DefaultContinuousVesselVolume; Kc is the same gain, 1/tau with
        tau a hundredth of a residence time by default.
    integral_time : float, optional
        Ti, in s. Defaults to 4 / Kc, which damps the level critically (the
        level is an integrating process: e'' + Kc e' + Kc / Ti e = 0).
    derivative_time : float, optional
        Td, in s. None (the default) for a PI loop.
    filter_time : float, optional
        Time constant of the filter on the volume the derivative reads.
        Defaults to Td / 10.
    tracking_time : float, optional
        Anti-windup tracking time constant. Defaults to Ti.
    """

    def __init__(
        self,
        target_volume=None,
        tau=None,
        K=None,
        integral_time=None,
        derivative_time=None,
        filter_time=None,
        tracking_time=None,
    ):

        super().__init__(target_volume=target_volume, tau=tau, K=K)

        self.integral_time = integral_time
        self.derivative_time = derivative_time
        self.filter_time = filter_time
        self.tracking_time = tracking_time

        self.level_loop = PIDLoop("level", action="direct",
                                  derivative=derivative_time is not None,
                                  units="m3")

        # Starting values of the loop's solver states (see
        # Controller.initial_solver_state).
        self.level_integral = 0.0
        self.level_filtered = None

    @property
    def solver_states(self):
        return tuple(super().solver_states) + self.level_loop.solver_states

    def initial_solver_state(self, state_key, unit=None):

        if state_key == self.level_loop.filtered_key:
            # The filter starts at the measurement, so the derivative starts
            # at zero.
            if self.level_filtered is None and unit is not None:
                return unit.Phases.vol

            return self.level_filtered

        return super().initial_solver_state(state_key, unit)

    def observe(self, time, completed_state, unit, resolved_inlets=None,
                resolved_outlets=None):

        super().observe(time, completed_state, unit, resolved_inlets,
                        resolved_outlets)

        if (self.level_loop.filtered_key is not None
                and self.level_filtered is None):
            self.level_filtered = self.level_loop.initial_filtered(
                unit.Phases.vol)

    def level_tuning(self, inlet_flow):
        """Kc, Ti, Td and Tf at this inlet flow."""

        gain = self.get_gain(inlet_flow)
        integral_time = (4.0 / gain if self.integral_time is None
                         else self.integral_time)
        derivative_time = self.derivative_time
        filter_time = self.filter_time

        if derivative_time is not None and filter_time is None:
            filter_time = 0.1 * derivative_time

        return gain, integral_time, derivative_time, filter_time

    def requested_outlet_flow(self, unit, resolved_inlets=None,
                              completed_state=None):

        inlet_flow = self.total_inlet_flow(unit, resolved_inlets)

        target = self.target_volume

        if target is None:
            target = unit.Phases.vol

        if completed_state is None:
            completed_state = {}

        gain, integral_time, derivative_time, filter_time = (
            self.level_tuning(inlet_flow))

        state = dict(completed_state)
        state.setdefault(self.level_loop.integral_key, self.level_integral)

        if self.level_loop.filtered_key is not None:
            state.setdefault(self.level_loop.filtered_key,
                             unit.Phases.vol if self.level_filtered is None
                             else self.level_filtered)

        return self.level_loop.law(
            state, target, unit.Phases.vol, gain, integral_time,
            derivative_time, filter_time, bias=inlet_flow)

    def actuate(self, time, completed_state, unit, resolved_inlets=None,
                resolved_outlets=None):

        if resolved_inlets is None:
            return

        inlet_flow = self.total_inlet_flow(unit, resolved_inlets)

        self.operating_conditions[
            OperatingKey("vol_flow", connection=0, port='outlet')
        ] = self.soft_floor(
            self.requested_outlet_flow(unit, resolved_inlets,
                                       completed_state),
            self.floor_width * np.maximum(inlet_flow, eps),
        )

    def get_solver_state_rates(self, time, completed_state, unit,
                               resolved_inlets=None, resolved_outlets=None,
                               operating_conditions=None):

        rates = dict(super().get_solver_state_rates(
            time, completed_state, unit, resolved_inlets, resolved_outlets,
            operating_conditions))

        inlet_flow = self.total_inlet_flow(unit, resolved_inlets)
        law = self.requested_outlet_flow(unit, resolved_inlets,
                                         completed_state)
        gain, integral_time, _, filter_time = self.level_tuning(inlet_flow)

        rates.update(self.level_loop.rates(
            completed_state, self.target_volume, unit.Phases.vol, law,
            self.applied_outlet_flow(resolved_outlets, law), gain,
            integral_time, filter_time, self.tracking_time))

        return rates

    @staticmethod
    def applied_outlet_flow(resolved_outlets, law):
        """The outflow that actually left, after the floor and the limiter.

        That, not the requested flow, is what the integral must track: an
        outlet held shut by the floor or throttled to protect an inventory
        would otherwise wind the integral up.
        """

        if not resolved_outlets:
            return law

        return sum(
            transfer.vol_flow
            for connection in resolved_outlets
            for transfer in connection
        )


class PIDTemperatureMixin:
    """
    Track a temperature profile with a PI(D) loop on the jacket.

        T_jacket = T_set + Kc * (e + I / Ti + Td * dT_f/dt),   e = T_set - T

    clipped to ``jacket_limits``, and applied as the utility's inlet
    temperature (MultiPhaseVessel.UTILITY_TEMPERATURE). The set point is the
    bias, so a vessel with no other heat load needs no integral at all.

    Unlike TemperatureProfileMixin, which imposes the profile and replaces
    the energy balance, this keeps the energy balance and acts through the
    jacket. The temperature follows the profile as well as the jacket can
    make it, so the vessel needs a Utility, and must not be isothermal or
    adiabatic.

    Default tuning, from the vessel at its first evaluation: the jacket time
    constant tau_ht = m cp / (U A); Ti = tau_ht and Kc = tau_ht / lambda with
    the closed-loop time constant lambda = tau_ht / 4 (IMC tuning of a PI
    loop for a first-order process of unit gain), so Kc = 4.
    """

    def set_temperature_loop(self, temp_func, gain=None, integral_time=None,
                             derivative_time=None, filter_time=None,
                             tracking_time=None, jacket_limits=(None, None),
                             closed_loop_fraction=0.25):

        self.temp_func = temp_func
        self.temperature_gain = gain
        self.temperature_integral_time = integral_time
        self.temperature_derivative_time = derivative_time
        self.temperature_filter_time = filter_time
        self.temperature_tracking_time = tracking_time
        self.jacket_limits = tuple(jacket_limits)
        self.closed_loop_fraction = closed_loop_fraction

        self.temperature_loop = PIDLoop(
            "temperature", action="reverse",
            derivative=derivative_time is not None, units="K")

        self.temperature_integral = 0.0
        self.temperature_filtered = None
        self.heat_transfer_time = None

    @property
    def solver_states(self):
        return tuple(super().solver_states) + self.temperature_loop.solver_states

    def initial_solver_state(self, state_key, unit=None):

        if state_key == self.temperature_loop.filtered_key:
            if self.temperature_filtered is None and unit is not None:
                return unit.get_heat_transfer_temperature()

            return self.temperature_filtered

        return super().initial_solver_state(state_key, unit)

    def observe(self, time, completed_state, unit, resolved_inlets=None,
                resolved_outlets=None):

        super().observe(time, completed_state, unit, resolved_inlets,
                        resolved_outlets)

        if self.heat_transfer_time is None:

            if unit.Utility is None or not unit.has_utility_balance:
                raise ValueError(
                    f"{type(self).__name__} acts through the jacket, so the "
                    "vessel needs a Utility and an energy balance (not "
                    "isothermal, not adiabatic).")

            basis = 'mass' if unit.basis == 'mass_j' else unit.basis
            heat_capacity = unit.Phases.getCp(basis=basis)
            conductance = unit.u_ht * unit.get_heat_transfer_area()
            self.heat_transfer_time = float(heat_capacity / conductance)

        if (self.temperature_loop.filtered_key is not None
                and self.temperature_filtered is None):
            self.temperature_filtered = self.temperature_loop.initial_filtered(
                self.measured_temperature(completed_state, unit))

    def temperature_tuning(self):
        """Kc, Ti, Td and Tf of the temperature loop."""

        tau = self.heat_transfer_time
        gain = (1.0 / self.closed_loop_fraction
                if self.temperature_gain is None else self.temperature_gain)
        integral_time = (tau if self.temperature_integral_time is None
                         else self.temperature_integral_time)
        derivative_time = self.temperature_derivative_time
        filter_time = self.temperature_filter_time

        if derivative_time is not None and filter_time is None:
            filter_time = 0.1 * derivative_time

        return gain, integral_time, derivative_time, filter_time

    @staticmethod
    def measured_temperature(completed_state, unit):
        temp = completed_state.get(StateKey("global_temp"))
        return unit.get_heat_transfer_temperature() if temp is None else temp

    def jacket_temperature(self, time, completed_state, unit):
        """(the law's jacket temperature, the one applied after the limits)"""

        setpoint = np.of_time(self.temp_func, time)
        measurement = self.measured_temperature(completed_state, unit)
        gain, integral_time, derivative_time, filter_time = (
            self.temperature_tuning())

        state = dict(completed_state)
        state.setdefault(self.temperature_loop.integral_key,
                         self.temperature_integral)

        if self.temperature_loop.filtered_key is not None:
            state.setdefault(self.temperature_loop.filtered_key,
                             measurement if self.temperature_filtered is None
                             else self.temperature_filtered)

        law = self.temperature_loop.law(
            state, setpoint, measurement, gain, integral_time,
            derivative_time, filter_time, bias=setpoint)

        low, high = self.jacket_limits
        applied = law

        if low is not None:
            applied = np.maximum(applied, low)

        if high is not None:
            applied = np.minimum(applied, high)

        return law, applied

    def actuate(self, time, completed_state, unit, resolved_inlets=None,
                resolved_outlets=None):

        super().actuate(time, completed_state, unit, resolved_inlets,
                        resolved_outlets)

        # On the first pass, which sets everything that does not depend on
        # the resolved inlets.
        if resolved_inlets is None:
            _, applied = self.jacket_temperature(time, completed_state, unit)
            self.operating_conditions[unit.UTILITY_TEMPERATURE] = applied

    def get_solver_state_rates(self, time, completed_state, unit,
                               resolved_inlets=None, resolved_outlets=None,
                               operating_conditions=None):

        rates = dict(super().get_solver_state_rates(
            time, completed_state, unit, resolved_inlets, resolved_outlets,
            operating_conditions))

        law, applied = self.jacket_temperature(time, completed_state, unit)
        gain, integral_time, _, filter_time = self.temperature_tuning()

        rates.update(self.temperature_loop.rates(
            completed_state, np.of_time(self.temp_func, time),
            self.measured_temperature(completed_state, unit), law, applied,
            gain, integral_time, filter_time,
            self.temperature_tracking_time))

        return rates


class PIDTemperatureController(PIDTemperatureMixin, Controller):
    """
    Track a temperature profile through the jacket with a PI(D) loop.

    The feedback counterpart of SimpleTemperatureController, which imposes
    the profile on the vessel directly. See PIDTemperatureMixin for the law
    and its default tuning.
    """

    def __init__(self, temp_func, gain=None, integral_time=None,
                 derivative_time=None, filter_time=None, tracking_time=None,
                 jacket_limits=(None, None), closed_loop_fraction=0.25):

        super().__init__()

        self.set_temperature_loop(
            temp_func, gain=gain, integral_time=integral_time,
            derivative_time=derivative_time, filter_time=filter_time,
            tracking_time=tracking_time, jacket_limits=jacket_limits,
            closed_loop_fraction=closed_loop_fraction)


class PIDContinuousVesselController(TemperatureProfileMixin,
                                    PIDContinuousVesselVolume):
    """
    ContinuousVesselController with a PI(D) volume loop.

    The drop-in counterpart of ContinuousVesselController: the temperature
    is imposed on ``temp_func`` exactly as there (the energy balance is
    replaced, no Utility needed), and the volume is held by
    PIDContinuousVesselVolume, so exactly at target rather than with the
    proportional loop's offset.

    Parameters
    ----------
    temp_func, temp_0 :
        As in ContinuousVesselController.
    target_volume, tau, K, integral_time, derivative_time, filter_time,
    tracking_time :
        The volume loop, as in PIDContinuousVesselVolume.
    """

    def __init__(self, temp_func, temp_0=None, target_volume=None, tau=None,
                 K=None, integral_time=None, derivative_time=None,
                 filter_time=None, tracking_time=None):

        super().__init__(
            target_volume=target_volume, tau=tau, K=K,
            integral_time=integral_time, derivative_time=derivative_time,
            filter_time=filter_time, tracking_time=tracking_time)

        self.set_temperature_profile(temp_func, temp_0)


class JacketPIDContinuousVesselController(PIDTemperatureMixin,
                                          PIDContinuousVesselVolume):
    """
    PI(D) temperature through the jacket, PI(D) volume through the outlet.

    The full feedback version of ContinuousVesselController: one controller
    (one per vessel) running two loops, each with its own solver states. The
    volume is held exactly at target, and the temperature follows
    ``temp_func`` through the jacket rather than being imposed. The vessel
    therefore needs a Utility and keeps its energy balance.

    Parameters
    ----------
    temp_func : callable
        The temperature set point, ``temp_func(time) -> K``.
    target_volume, tau, K, integral_time, derivative_time, filter_time,
    tracking_time :
        The volume loop, as in PIDContinuousVesselVolume.
    temperature : dict, optional
        Keyword arguments of the temperature loop (gain, integral_time,
        derivative_time, filter_time, tracking_time, jacket_limits,
        closed_loop_fraction), as in PIDTemperatureController.
    """

    def __init__(self, temp_func, target_volume=None, tau=None, K=None,
                 integral_time=None, derivative_time=None, filter_time=None,
                 tracking_time=None, temperature=None):

        super().__init__(
            target_volume=target_volume, tau=tau, K=K,
            integral_time=integral_time, derivative_time=derivative_time,
            filter_time=filter_time, tracking_time=tracking_time)

        self.set_temperature_loop(temp_func, **(temperature or {}))


class TankLevelController(Controller):
    """
    Hold a level by switching the outlet fully on or fully off.

    A worked example of the minimal controller: override ``actuate`` and set
    entries in ``self.operating_conditions``. Note the hard switch at the
    target volume is a discontinuity in the right-hand side, so pair it with
    the matching event (see DefaultContinuousVesselVolume.get_events) or
    expect the integrator to hunt for it.
    """

    def __init__(self, target_volume=2.0):
        super().__init__()
        self.target_volume = target_volume

    def actuate(
        self,
        time,
        completed_state,
        unit,
        resolved_inlets=None,
        resolved_outlets=None,
    ):

        if resolved_inlets is None:
            return

        inlet_flow = sum(
            transfer.stream_phase.vol_flow
            for connection in resolved_inlets
            for transfer in connection
        )

        outlet_flow = (
            0.0
            if unit.Phases.vol < self.target_volume
            else inlet_flow
        )

        self.operating_conditions[
            OperatingKey(
                "vol_flow",
                connection=0,
                port="outlet",
            )
        ] = outlet_flow

    def get_events(self, unit):

        def level_crossing(time, completed_state, unit):
            return float(unit.Phases.vol) - self.target_volume

        return [
            StateEvent(
                name="tank_level",
                function=level_crossing,
                direction=0,
                terminal=False,
                source=self,
            )
        ]


class ComplexController(Controller):
    """
    Template for a controller that manipulates both ends of a vessel.

    There is only ever one controller per vessel, so anything driving several
    variables does all of it here. The inlet is set on the first pass, when
    ``resolved_inlets`` is still None, and the outlet on the second, once the
    actual inlet is known; ``MultiPhaseVessel.get_operating_conditions``
    merges the two passes.
    """

    def actuate(
        self,
        time,
        completed_state,
        unit,
        resolved_inlets=None,
        resolved_outlets=None,
    ):

        if resolved_inlets is None:
            self.actuate_inlets(time, completed_state, unit)
        else:
            self.actuate_outlets(
                time, completed_state, unit, resolved_inlets,
            )

    def actuate_inlets(self, time, completed_state, unit):
        """Set OperatingKey(..., port='inlet') entries here."""

    def actuate_outlets(self, time, completed_state, unit, resolved_inlets):
        """Set OperatingKey(..., port='outlet') entries here."""
