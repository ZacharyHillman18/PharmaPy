from PharmaPy.DataClasses import (PhaseConnection,PhaseMapping,PhaseRef,PhaseStateCollection,PhaseStateVariable,
                                  IntraPhaseProcess,StateCollection,StateKey,StateVariable,StreamConnection,
                                  OperatingKey, StateEvent)
from typing import Any
from PharmaPy import xp as np
eps = np.finfo(float).eps


class Controller:

    def __init__(self):
        self.states = {}
        self.operating_conditions = {}

    def reset(self):
        self.states.clear()
        self.operating_conditions.clear()

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

    def requested_outlet_flow(self, unit, resolved_inlets=None):
        """
        The outlet the loop asks for before the no-backflow floor.

        Shared by actuate and the switching event so both describe the same
        surface.
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

            return float(self.requested_outlet_flow(unit, resolved_inlets))

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
