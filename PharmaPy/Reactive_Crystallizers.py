"""
Reactive crystallizers: a reaction in the liquid and a population balance
in the solid, in one vessel.

Each class is a reactor and a crystallizer at once. The reactor half
(Reactors_Refactored._BaseReactor) contributes the ``RxnKinetics``
convenience initializer, which places a ReactionMechanism in the liquid; the
crystallizer half (Crystallizers_Refactored) contributes ``CrystKinetics``,
the population balance's wiring, the dilution of the crystal population by
a growing volume, the Krylov hint, and -- for the continuous one -- the
level controller and the default outlet. Nothing else is needed: both halves
build on MultiPhaseVessel, and the vessel already sums the reaction's
intraphase rates and the crystallization's cross-phase rates.

    unit = SemiBatchReactiveCrystallizer(controller=...)
    unit.Phases = [liquid, solid]          # the solid carries its FVM/moments
    unit.RxnKinetics = RxnKinetics(...)    # NaClO4 + NH4Cl --> AP + NaCl
    unit.CrystKinetics = CrystKinetics(...)
    unit.Inlet = feed

``RxnKinetics`` and ``CrystKinetics`` may be set in either order, after
``Phases``. For anything the convenience initializers do not cover, set
``intraphase_processes`` and ``phase_connections`` directly, as for the
reactors and crystallizers themselves.
"""

from PharmaPy.Crystallizers_Refactored import (BatchCrystallizer,
                                               ContinuousCrystallizer,
                                               SemiBatchCrystallizer)
from PharmaPy.Reactors_Refactored import _BaseReactor


class BatchReactiveCrystallizer(_BaseReactor, BatchCrystallizer):
    """Batch: reaction and crystallization, no feed, no outlet."""

    oper_mode = "Batch"


class SemiBatchReactiveCrystallizer(_BaseReactor, SemiBatchCrystallizer):
    """Semibatch: fed (a reactant, typically), no outlet. The crystal
    population is diluted as the feed grows the volume."""

    oper_mode = "Semibatch"


class ContinuousReactiveCrystallizer(_BaseReactor, ContinuousCrystallizer):
    """Continuous (a reactive MSMPR): fed, with the outlet holding the
    level (DefaultContinuousVesselVolume unless a controller is given)."""

    oper_mode = "Continuous"
