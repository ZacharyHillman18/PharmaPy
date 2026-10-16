"""ConcurrentVessel driver: several vessels integrated as one system.

Staged like FlowsheetTester and built from its units, so that a failure here
is attributable to the coupling rather than to the unit models:

    Stage 1  R01 -> R02 solved concurrently == the same pair through
             SimulationExec
    Stage 2  a recycle, converged inside the right-hand side: mass closes,
             levels hold, the recycled flow is the share asked for
    Stage 3  the same recycle with the members listed the other way round,
             so the loop is torn on the other link
    Stage 4  the recycle as algebraic states on a DAE backend == iterated
    Stage 5  iteration that cannot converge: refused on an ODE-only backend,
             restarted with algebraic tears when a DAE backend is supplied
    Stage 6  a controller on R01 acting on R02's concentration, and a
             replay that reports what it actually did
    Stage 7  crystallizer pairs, whose link carries a size distribution ==
             SimulationExec
    Stage 8  a ConcurrentVessel as one node of a SimulationExec flowsheet
    Stage 9  cascade control: R02's controller sets the level R01's
             controller holds
    Stage 10 a crystallizer recycle, whose tear carries a size
             distribution: exact recycle, either tear, iterated or
             algebraic

usage: python ConcurrentTester.py [assimulo|scipy]
"""

import os
import sys
import traceback
import warnings

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

# FlowsheetTester reads the backend from sys.argv, the same way this script
# does, so both agree on it.
import FlowsheetTester as FT  # noqa: E402
from FlowsheetTester import (PATH, TIME_R01, TEMP_INIT, BACKEND,  # noqa: E402
                             make_integrator, quiet, _new_reactor, _new_feed,
                             _continuous_reactor, _continuous_cryst,
                             _semibatch_cryst, NewContReactor,
                             NewSemiReactor, ContinuousVesselController)

from PharmaPy.SimExec import SimulationExec  # noqa: E402
from PharmaPy.ConcurrentVessel import (ConcurrentVessel, StreamLink,  # noqa: E402
                                       TearConvergenceError)
from PharmaPy.IntegratorBackends import ScipyBackend  # noqa: E402
from PharmaPy.DataClasses import OperatingKey, PhaseRef  # noqa: E402

try:
    from PharmaPy.IntegratorBackends import AssimuloDAEBackend
    AssimuloDAEBackend()
except Exception:  # Assimulo is optional
    AssimuloDAEBackend = None

RECYCLE = 0.5
RECYCLE_GRID = np.linspace(0.0, 3 * TIME_R01, 3001)


def rel(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    return float(np.max(np.abs(a - b)) / max(np.max(np.abs(b)), 1e-30))


def make_dae_integrator():
    if AssimuloDAEBackend is None:
        raise RuntimeError('this stage needs Assimulo (AssimuloDAEBackend)')

    return AssimuloDAEBackend(options={'maxh': 60})


def _fed_reactor():
    """A continuous reactor with nothing of its own coming in."""
    return _new_reactor(
        NewContReactor,
        controller=ContinuousVesselController(temp_func=lambda t: TEMP_INIT))


def _recycle_pair(fraction=RECYCLE, first='R01', **kwargs):
    """R01 (fresh feed) -> R02, with `fraction` of R02's outlet back to R01."""
    members = {'R01': _continuous_reactor(), 'R02': _fed_reactor()}

    if first == 'R02':
        members = {'R02': members['R02'], 'R01': members['R01']}

    links = [StreamLink('R01', 'R02'),
             StreamLink('R02', 'R01', fraction=fraction)]

    kwargs.setdefault('integrator', make_integrator())

    return ConcurrentVessel(members, links, **kwargs)


def _mass_closure(cv, fraction=RECYCLE):
    """
    |change in holdup - integral(feed - purge)|, as a share of the feed.

    The purge is what R02 discharges and the link does not take back, at
    R02's own density: the outlet leaves at the vessel's state.
    """
    r1, r2 = cv.R01, cv.R02
    t = np.asarray(cv.result.time)

    held = (np.asarray(r1.result.Total_m_in_vessel)
            + np.asarray(r2.result.Total_m_in_vessel))
    density = (np.asarray(r2.result.Total_m_in_vessel)
               / np.asarray(r2.result.vessel_vol))
    purge = (1 - fraction) * np.ravel(r2.result.outlet_vol_flow) * density
    feed = float(_new_feed().mass_flow)

    gap = (held[-1] - held[0]) - np.trapezoid(feed - purge, t)

    return abs(gap) / (feed * t[-1])


_CACHE = {}


def _iterated_recycle():
    """Stage 2's system, solved once and shared with the stages after it."""
    if 'iterated' not in _CACHE:
        cv = _recycle_pair()
        cv.solve_unit(time_grid=RECYCLE_GRID, verbose=False)
        _CACHE['iterated'] = cv

    return _CACHE['iterated']


def _final_gap(cv, reference):
    return max(rel(np.asarray(cv[name].result.mole_conc_liquid0)[-1],
                   np.asarray(reference[name].result.mole_conc_liquid0)[-1])
               for name in ('R01', 'R02'))


# =====================================================================
# Stage 1 -- with no recycle the concurrent solve is the sequential one,
# minus the interpolation of the upstream trajectory.
# =====================================================================
def stage1_acyclic_matches_simexec():
    flst = SimulationExec(PATH, flowsheet='R01 --> R02')
    flst.R01 = _continuous_reactor()
    flst.R02 = _continuous_reactor()
    flst.SolveFlowsheet(kwargs_run=quiet({'R01': {'runtime': TIME_R01},
                                          'R02': {'runtime': TIME_R01}}),
                        verbose=False)

    r1, r2 = _continuous_reactor(), _continuous_reactor()

    # inlet=0 replaces R02's placeholder feed, as Connections does.
    cv = ConcurrentVessel({'R01': r1, 'R02': r2},
                          [StreamLink('R01', 'R02', inlet=0)],
                          integrator=make_integrator())
    cv.solve_unit(runtime=TIME_R01, verbose=False)

    if cv.tears:
        raise AssertionError('an acyclic pair was given a tear: %s' % cv.tears)

    if len(r2.inlet_connections) != 1:
        raise AssertionError('inlet=0 added a connection instead of '
                             'replacing the placeholder feed')

    gaps = {}

    for name, ours, theirs in (('R01', r1, flst.R01), ('R02', r2, flst.R02)):
        gaps[name] = rel(np.asarray(ours.result.mole_conc_liquid0)[-1],
                         np.asarray(theirs.result.mole_conc_liquid0)[-1])

        if gaps[name] > 1e-5:
            raise AssertionError('%s differs from SimulationExec by %.3e'
                                 % (name, gaps[name]))

    # What R02 received is what R01 reports it discharged.
    received = cv.result.couplings['R01->R02']['vol_flow']
    sent = np.ravel(r1.result.outlet_vol_flow)
    handover = rel(received, sent)

    if handover > 1e-12:
        raise AssertionError('R02 received %.3e less/more than R01 reports '
                             'discharging' % handover)

    return ('R01 %.1e, R02 %.1e from SimulationExec; handover exact (%.0e)'
            % (gaps['R01'], gaps['R02'], handover))


# =====================================================================
# Stage 2 -- a recycle, iterated inside every right-hand side evaluation.
# =====================================================================
def stage2_recycle_iterated():
    cv = _iterated_recycle()

    if [link.label for link in cv.tears] != ['R02->R01']:
        raise AssertionError('expected the loop torn on R02->R01, got %s'
                             % [link.label for link in cv.tears])

    if cv.requires_dae or cv.has_algebraic_balance:
        raise AssertionError('an iterated recycle should stay an ODE')

    # The bound is set to catch a leak, not integration error. A mis-scaled
    # recycle share leaks 1e-3 of the feed or more. Integration error alone
    # is 2e-7 on scipy, but 1.3e-5 on CVode at its default tolerances: CVode
    # carries a ~0.5% error in the outlet flows through the first seconds of
    # the coupled start-up, which closes once the transient has decayed, and
    # at rtol=1e-9 it agrees with a tight reference to 5e-8.
    closure = _mass_closure(cv)

    if closure > 1e-4:
        raise AssertionError('mass does not close: %.3e of the feed' % closure)

    for name in ('R01', 'R02'):
        vol = np.asarray(cv[name].result.vessel_vol)
        drift = abs(vol[-1] - vol[0]) / vol[0]

        if drift > 1e-3:
            raise AssertionError('%s level drifted %.2e' % (name, drift))

    recycled = cv.result.couplings['R02->R01']['vol_flow']
    discharged = np.ravel(cv.R02.result.outlet_vol_flow)
    share = rel(recycled, RECYCLE * discharged)

    if share > 1e-12:
        raise AssertionError('the recycle is not %.0f%% of R02 outlet '
                             '(rel %.3e)' % (100 * RECYCLE, share))

    stats = cv.tear_statistics

    return ('mass closes to %.1e of feed; %.1f sweeps per evaluation, at '
            'most %d' % (closure, stats['sweeps'] / stats['evaluations'],
                         stats['max_sweeps']))


# =====================================================================
# Stage 3 -- the tear is an artefact of evaluation order, not of the
# physics: listing the members the other way tears the other link and must
# give the same answer.
# =====================================================================
def stage3_tear_choice_is_immaterial():
    reference = _iterated_recycle()

    cv = _recycle_pair(first='R02')
    cv.solve_unit(time_grid=RECYCLE_GRID, verbose=False)

    if [link.label for link in cv.tears] != ['R01->R02']:
        raise AssertionError('expected the loop torn on R01->R02, got %s'
                             % [link.label for link in cv.tears])

    gap = _final_gap(cv, reference)

    if gap > 1e-5:
        raise AssertionError('tearing the other link moved the answer by '
                             '%.3e' % gap)

    return 'torn on R01->R02 instead, final state within %.1e' % gap


# =====================================================================
# Stage 4 -- the recycle as algebraic states, solved by the integrator.
# =====================================================================
def stage4_recycle_algebraic():
    reference = _iterated_recycle()

    cv = _recycle_pair(tear_mode='algebraic',
                       integrator=make_dae_integrator())
    cv.solve_unit(time_grid=RECYCLE_GRID, verbose=False)

    if not cv.requires_dae:
        raise AssertionError('algebraic tears did not set requires_dae')

    alg = [name for name, di in cv.states_di.items() if di['type'] == 'alg']

    if not alg or not all(name.startswith('tear[') for name in alg):
        raise AssertionError('expected only tear states to be algebraic, '
                             'got %s' % alg)

    closure = _mass_closure(cv)

    if closure > 1e-4:
        raise AssertionError('mass does not close: %.3e of the feed' % closure)

    gap = _final_gap(cv, reference)

    if gap > 1e-5:
        raise AssertionError('algebraic and iterated recycles differ by %.3e'
                             % gap)

    return ('%d algebraic tear states; mass closes to %.1e; matches the '
            'iterated recycle to %.1e' % (len(alg), closure, gap))


# =====================================================================
# Stage 5 -- iteration that cannot converge (one sweep allowed).
# =====================================================================
def stage5_fallback_to_algebraic():
    reference = _iterated_recycle()

    # (a) Nothing to fall back on: refuse, saying a DAE backend is needed.
    cv = _recycle_pair(integrator=ScipyBackend(options={'maxh': 60}),
                       max_tear_sweeps=1)

    try:
        cv.solve_unit(time_grid=RECYCLE_GRID, verbose=False)
    except TearConvergenceError as exc:
        if 'DAE' not in str(exc) or not cv.requires_dae:
            raise AssertionError('refused without saying a DAE backend is '
                                 'needed: %s' % exc) from None
    else:
        raise AssertionError('a recycle that cannot converge in one sweep '
                             'was integrated as an ODE anyway')

    # (b) A DAE backend supplied for the fallback: warn, switch, finish.
    cv = _recycle_pair(max_tear_sweeps=1,
                       dae_integrator=make_dae_integrator())

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        cv.solve_unit(time_grid=RECYCLE_GRID, verbose=False)

    warned = [w for w in caught if issubclass(w.category, RuntimeWarning)
              and 'algebraic' in str(w.message)]

    if not warned:
        raise AssertionError('fell back without warning')

    if cv.active_tear_mode != 'algebraic' or not cv.requires_dae:
        raise AssertionError('did not record the switch to algebraic tears')

    if cv.tear_fallback is None:
        raise AssertionError('tear_fallback does not say why it switched')

    gap = _final_gap(cv, reference)

    if gap > 1e-5:
        raise AssertionError('the fallback solve differs from the iterated '
                             'one by %.3e' % gap)

    return ('ODE-only backend refused; with a DAE fallback it switched at '
            't=%g and matches to %.1e' % (cv.tear_fallback['time'], gap))


# =====================================================================
# Stage 6 -- control across units. R01's single controller sets R01's feed
# from the concentration of C measured in R02, downstream of it.
# =====================================================================
class FeedFromDownstream(ContinuousVesselController):
    """Proportional feed control on a downstream composition."""

    def __init__(self, downstream, c_set, gain, nominal_flow):
        super().__init__(temp_func=lambda t: TEMP_INIT)

        self.downstream = downstream
        self.c_set = c_set
        self.gain = gain
        self.nominal_flow = nominal_flow

    def actuate(self, time, completed_state, unit, resolved_inlets=None,
                resolved_outlets=None):

        if resolved_inlets is None:
            measured = float(self.downstream.Phases.Liquids[0].mole_conc[2])
            error = (self.c_set - measured) / self.c_set
            flow = self.nominal_flow * max(1.0 + self.gain * error, 0.05)

            key = OperatingKey('vol_flow', connection=0,
                               phaseref=PhaseRef('liquid', 0), port='inlet')
            self.operating_conditions[key] = flow
            return

        super().actuate(time, completed_state, unit, resolved_inlets,
                        resolved_outlets)


def _open_loop_c():
    """R02's [C] after 3 h with nothing controlled, cached."""
    if 'c_open' not in _CACHE:
        open_loop = ConcurrentVessel(
            {'R01': _continuous_reactor(), 'R02': _fed_reactor()},
            [StreamLink('R01', 'R02')], integrator=make_integrator())
        open_loop.solve_unit(runtime=3 * TIME_R01, verbose=False)

        _CACHE['c_open'] = float(
            np.asarray(open_loop.R02.result.mole_conc_liquid0)[-1, 2])

    return _CACHE['c_open']


def _feed_controlled_pair(c_set, order=None):
    r2 = _fed_reactor()
    r1 = _new_reactor(NewContReactor, inlet=_new_feed(),
                      controller=FeedFromDownstream(r2, c_set, gain=8.0,
                                                    nominal_flow=FT.FEED_VOLFLOW))

    cv = ConcurrentVessel({'R01': r1, 'R02': r2}, [StreamLink('R01', 'R02')],
                          integrator=make_integrator(), order=order)
    cv.solve_unit(time_grid=np.linspace(0, 3 * TIME_R01, 31), verbose=False)

    return cv


def _order_gap(cv, other):
    """How far two solves of one system, evaluated in different orders, end."""
    return max(
        rel(np.asarray(cv[name].result.mass_j_liquid0)[-1],
            np.asarray(other[name].result.mass_j_liquid0)[-1])
        for name in cv.members)


def stage6_control_across_units():
    c_open = _open_loop_c()
    c_set = 1.1 * c_open

    cv = _feed_controlled_pair(c_set)
    r1, r2 = cv.R01, cv.R02

    c_closed = float(np.asarray(r2.result.mole_conc_liquid0)[-1, 2])

    if abs(c_closed - c_set) >= abs(c_open - c_set):
        raise AssertionError('the controller did not move R02 towards its set '
                             'point: open %.5f, closed %.5f, set %.5f'
                             % (c_open, c_closed, c_set))

    # The replay must see what the controller saw during the run. Replaying
    # each member on its own handed R01's controller a frozen copy of R02,
    # and R01's reported outlet was then wrong by up to 85%.
    reported = np.ravel(r1.result.outlet_vol_flow)
    carried = cv.result.couplings['R01->R02']['vol_flow']
    replay = rel(reported, carried)

    if replay > 1e-10:
        raise AssertionError('R01 reports an outlet flow %.3e away from what '
                             'R02 received' % replay)

    # R01's controller reads R02 while R01 is being evaluated. Evaluating R02
    # first instead (which makes R01 -> R02 a tear) must not change what it
    # reads: every member is brought to the evaluated state before any
    # balance runs, so the measurement is a function of the state alone.
    reordered = _feed_controlled_pair(c_set, order=['R02', 'R01'])

    if not reordered.tears:
        raise AssertionError('evaluating R02 first should have torn R01 -> R02')

    order_gap = _order_gap(cv, reordered)

    if order_gap > 1e-6:
        raise AssertionError('evaluating R02 first moved the controlled '
                             'result by %.3e' % order_gap)

    return ('R02 [C] %.5f -> %.5f towards %.5f; replay consistent to %.0e; '
            'evaluation order immaterial (%.0e)'
            % (c_open, c_closed, c_set, replay, order_gap))


# =====================================================================
# Stage 9 -- cascade control. R02's controller is the outer loop: from
# R02's [C] it decides what level R01 should hold. R01's controller is the
# inner loop: an ordinary level controller whose set point is that request.
#
# The set point is PULLED: R01's controller asks the master whenever it
# needs it, and the master answers from R02's current state. Having the
# master write into R01's controller instead would make the answer depend on
# which of the two happened to be evaluated last.
# =====================================================================
class QualityMaster(ContinuousVesselController):
    """R02's controller, which is also the outer loop of a cascade."""

    def __init__(self, c_set, base_volume, span=0.3, gain=10.0):
        super().__init__(temp_func=lambda t: TEMP_INIT)

        self.c_set = c_set
        self.base_volume = base_volume
        self.span = span
        self.gain = gain

        # The vessel this controller runs, assigned once it is built.
        self.vessel = None

    def level_setpoint(self):
        """The volume R01 should hold, from R02's [C] now."""

        return self.setpoint_for(self.vessel.Phases.Liquids[0].mole_conc[2])

    def setpoint_for(self, measured):
        """The outer loop's law; vectorised, so a recorded run can be fed in."""

        error = (self.c_set - np.asarray(measured, dtype=float)) / self.c_set

        # Residence time in R01 turns C on into D, so a shortfall of C asks
        # for LESS volume upstream. tanh keeps the request within span of the
        # base volume, smoothly, from the C-free start onwards.
        return self.base_volume * (1.0 - self.span * np.tanh(self.gain * error))


class LevelFollower(ContinuousVesselController):
    """R01's controller: a level loop whose set point the master owns."""

    def __init__(self, master):
        self.master = master
        super().__init__(temp_func=lambda t: TEMP_INIT)

    @property
    def target_volume(self):
        return self.master.level_setpoint()

    @target_volume.setter
    def target_volume(self, value):
        # Written by DefaultContinuousVesselVolume's constructor and by
        # observe. The set point is the master's, so neither is kept.
        pass


def _cascade(c_set, order=None):
    master = QualityMaster(c_set, base_volume=FT.VOL_INIT)

    r2 = _new_reactor(NewContReactor, controller=master)
    master.vessel = r2

    r1 = _new_reactor(NewContReactor, inlet=_new_feed(),
                      controller=LevelFollower(master))

    cv = ConcurrentVessel({'R01': r1, 'R02': r2}, [StreamLink('R01', 'R02')],
                          integrator=make_integrator(), order=order)
    cv.solve_unit(time_grid=np.linspace(0, 3 * TIME_R01, 31), verbose=False)

    return cv


def stage9_cascade_control():
    c_open = _open_loop_c()
    c_set = 1.02 * c_open

    cv = _cascade(c_set)
    r1, r2 = cv.R01, cv.R02
    master = r2.controller

    # What the outer loop asked for at every reported time, rebuilt from
    # R02's recorded [C] with the master's own law, against what R01 held.
    requested = master.setpoint_for(
        np.asarray(r2.result.mole_conc_liquid0)[:, 2])
    held = np.ravel(r1.result.vessel_vol)

    excursion = float(np.max(np.abs(requested - FT.VOL_INIT)))

    if excursion < 0.01 * FT.VOL_INIT:
        raise AssertionError('the outer loop never moved R01 set point '
                             '(%.2e m3 at most)' % excursion)

    # The inner loop is a proportional level loop with a time constant of
    # about a minute, so it trails a moving set point. Early on it trails by
    # far more: when the outer loop raises the set point, R01 can only fill
    # as fast as it is fed, since its outlet cannot run backwards. Once that
    # start-up is over it must stay within a small share of how far the set
    # point travelled.
    settled = np.asarray(r1.result.time) > 0.25 * np.asarray(r1.result.time)[-1]
    tracking = float(np.max(np.abs(held - requested)[settled])) / excursion

    if tracking > 0.05:
        raise AssertionError('the inner loop strays %.1f%% of the set point '
                             'excursion from what the outer loop asks for'
                             % (100 * tracking))

    requested = float(requested[-1])

    c_closed = float(np.asarray(r2.result.mole_conc_liquid0)[-1, 2])

    if abs(c_closed - c_set) >= abs(c_open - c_set):
        raise AssertionError('the cascade did not move R02 towards its set '
                             'point: open %.5f, closed %.5f, set %.5f'
                             % (c_open, c_closed, c_set))

    # R01's outlet is the inner loop's output. If the replay had handed the
    # follower a stale master, it would report a different outlet than R02
    # received during the run.
    replay = rel(np.ravel(r1.result.outlet_vol_flow),
                 cv.result.couplings['R01->R02']['vol_flow'])

    if replay > 1e-10:
        raise AssertionError('R01 reports an outlet flow %.3e away from what '
                             'R02 received' % replay)

    reordered = _cascade(c_set, order=['R02', 'R01'])
    order_gap = _order_gap(cv, reordered)

    if order_gap > 1e-6:
        raise AssertionError('evaluating R02 first moved the cascade result '
                             'by %.3e' % order_gap)

    return ('R01 set point %.4f -> %.4f m3, followed within %.1f%% of its '
            'excursion; R02 [C] %.5f -> %.5f towards %.5f; order immaterial '
            '(%.0e)' % (FT.VOL_INIT, requested, 100 * tracking, c_open,
                        c_closed, c_set, order_gap))


# =====================================================================
# Stage 7 -- a slurry link: the coupling carries a crystal size
# distribution, owned by a mechanism rather than by the phase.
# =====================================================================
def _cryst_pair_gap(make_down, label):
    flst = SimulationExec(PATH, flowsheet='U01 --> U02')
    flst.U01 = _continuous_cryst()
    flst.U02 = make_down()
    flst.SolveFlowsheet(kwargs_run=quiet({'U01': {'runtime': TIME_R01},
                                          'U02': {'runtime': TIME_R01}}),
                        verbose=False)

    up, down = _continuous_cryst(), make_down()
    cv = ConcurrentVessel({'U01': up, 'U02': down},
                          [StreamLink('U01', 'U02', inlet=0)],
                          integrator=make_integrator())
    cv.solve_unit(runtime=TIME_R01, verbose=False)

    carried = cv.result.couplings['U01->U02']

    if 'distrib' not in carried:
        raise AssertionError('%s: the link carried no distribution (%s)'
                             % (label, sorted(carried)))

    liquid = rel(np.asarray(down.result.mass_j_liquid0)[-1],
                 np.asarray(flst.U02.result.mass_j_liquid0)[-1])
    distrib = rel(np.asarray(down.result.distrib_solid0)[-1],
                  np.asarray(flst.U02.result.distrib_solid0)[-1])

    return liquid, distrib


CRYST_TOL = 1e-3


def stage7_crystallizer_pairs():
    notes = []

    for label, make_down in (('continuous -> continuous', _continuous_cryst),
                             ('continuous -> semibatch', _semibatch_cryst)):
        liquid, distrib = _cryst_pair_gap(make_down, label)

        if max(liquid, distrib) > CRYST_TOL:
            raise AssertionError(
                '%s: differs from SimulationExec, liquid %.3e, distribution '
                '%.3e' % (label, liquid, distrib))

        notes.append('%s liquid %.1e distrib %.1e' % (label, liquid, distrib))

    return '; '.join(notes)


# =====================================================================
# Stage 10 -- a crystallizer recycle: half of CR02's slurry goes back to
# CR01, so the tear carries a size distribution as well as a flow, and both
# members nucleate from a crystal-free start while it circulates.
# =====================================================================
CRYST_RECYCLE = 0.5
CRYST_RUN = TIME_R01 / 3


def _cryst_recycle(first='CR01', **kwargs):
    members = {'CR01': _continuous_cryst(), 'CR02': _continuous_cryst()}

    if first == 'CR02':
        members = {'CR02': members['CR02'], 'CR01': members['CR01']}

    links = [StreamLink('CR01', 'CR02', inlet=0),
             StreamLink('CR02', 'CR01', fraction=CRYST_RECYCLE)]

    kwargs.setdefault('integrator', make_integrator())

    cv = ConcurrentVessel(members, links, **kwargs)
    cv.solve_unit(runtime=CRYST_RUN, verbose=False)

    return cv


def _cryst_gap(cv, reference):
    """Largest relative difference in any member's final liquid or crystals."""
    gaps = []

    for name in ('CR01', 'CR02'):
        for field in ('mass_j_liquid0', 'distrib_solid0'):
            gaps.append(rel(np.asarray(getattr(cv[name].result, field))[-1],
                            np.asarray(getattr(reference[name].result,
                                               field))[-1]))

    return max(gaps)


CRYST_RECYCLE_TOL = 1e-3


def stage10_crystallizer_recycle():
    cv = _cryst_recycle()
    cr01, cr02 = cv.CR01, cv.CR02

    carried = cv.result.couplings['CR02->CR01']

    if 'distrib' not in carried:
        raise AssertionError('the recycle carried no size distribution (%s)'
                             % sorted(carried))

    # CR01 receives CR02's crystals as CR02 holds them -- a density per m3 of
    # slurry, which a split does not change -- at the share asked for.
    held = np.maximum(np.asarray(cr02.result.distrib_solid0), 0.0)
    distrib_gap = rel(carried['distrib'], held)

    # Over the whole run, start-up included: while nucleation starts the
    # vessel's positivity limiter throttles the outlet, and CR02's reported
    # flow must be the throttled one the link carried, not the level
    # controller's request (tests/test_outlet_flow_reporting.py).
    flow_gap = rel(carried['vol_flow'],
                   CRYST_RECYCLE * np.ravel(cr02.result.outlet_vol_flow))

    if max(distrib_gap, flow_gap) > 1e-12:
        raise AssertionError('the recycle is not CR02 outlet at %.0f%%: '
                             'distribution %.3e, flow %.3e'
                             % (100 * CRYST_RECYCLE, distrib_gap, flow_gap))

    if not np.any(carried['distrib'][-1] > 0):
        raise AssertionError('no crystals were recycled')

    reordered = _cryst_recycle(first='CR02')

    if [link.label for link in reordered.tears] != ['CR01->CR02']:
        raise AssertionError('expected the loop torn on CR01->CR02, got %s'
                             % [link.label for link in reordered.tears])

    order_gap = _cryst_gap(reordered, cv)

    if order_gap > CRYST_RECYCLE_TOL:
        raise AssertionError('tearing the other link moved the crystallizers '
                             'by %.3e' % order_gap)

    notes = ['recycle exact (%.0e); other tear within %.1e'
             % (max(distrib_gap, flow_gap), order_gap)]

    if AssimuloDAEBackend is not None:
        algebraic = _cryst_recycle(tear_mode='algebraic',
                                   integrator=make_dae_integrator())

        tear_states = sum(1 for di in algebraic.states_di.values()
                          if di['type'] == 'alg')
        alg_gap = _cryst_gap(algebraic, cv)

        if alg_gap > CRYST_RECYCLE_TOL:
            raise AssertionError('algebraic and iterated crystallizer '
                                 'recycles differ by %.3e' % alg_gap)

        notes.append('algebraic (%d tear fields) within %.1e'
                     % (tear_states, alg_gap))

    stats = cv.tear_statistics
    notes.append('%.1f sweeps per evaluation'
                 % (stats['sweeps'] / stats['evaluations']))

    return '; '.join(notes)


# =====================================================================
# Stage 8 -- a ConcurrentVessel as one node of a sequential flowsheet:
#     R00 --> [R01 <-> R02] --> R03
# =====================================================================
def stage8_flowsheet_node():
    flst = SimulationExec(PATH, flowsheet='R00 --> CV01 --> R03')
    flst.R00 = _continuous_reactor()
    flst.CV01 = _recycle_pair()
    flst.R03 = _new_reactor(NewSemiReactor, inlet=_new_feed())

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        flst.SolveFlowsheet(
            kwargs_run=quiet({name: {'runtime': TIME_R01}
                              for name in ('R00', 'CV01', 'R03')}),
            verbose=False)

    missing = [str(w.message) for w in caught
               if 'modeling objects' in str(w.message)]

    if missing:
        raise AssertionError('modeling-object check complained: %s'
                             % missing[0])

    cv = flst.CV01
    r1 = cv.R01

    # The flowsheet feed replaced R01's own feed and left the recycle alone.
    feed_stream = r1.inlet_connections[0].stream
    traj = getattr(feed_stream, 'y_upstream', None)

    if traj is None:
        traj = next((getattr(p, 'y_upstream', None) for p in feed_stream
                     if getattr(p, 'y_upstream', None) is not None), None)

    if traj is None:
        raise AssertionError("R01's flowsheet inlet carries no trajectory "
                             "from R00")

    if len(r1.inlet_connections) != 2:
        raise AssertionError('R01 should keep its recycle inlet beside the '
                             'flowsheet feed, has %d inlets'
                             % len(r1.inlet_connections))

    # R03 receives the product: R02's outlet less what goes back to R01.
    stream = flst.R03.inlet_connections[0].stream
    arrived = getattr(stream, 'y_inlet', None)

    if not arrived:
        arrived = next((p.y_inlet for p in stream
                        if getattr(p, 'y_inlet', None)), None)

    if not arrived or 'vol_flow' not in arrived:
        raise AssertionError('R03 received no feed rate from CV01')

    product = (1 - RECYCLE) * np.ravel(cv.R02.result.outlet_vol_flow)
    gap = rel(arrived['vol_flow'], product)

    if gap > 1e-12:
        raise AssertionError('R03 is fed %.3e away from the product share of '
                             "R02's outlet" % gap)

    if flst.result is None or 'CV01' not in repr(flst.result):
        raise AssertionError('the flowsheet summary does not list CV01')

    return ('fed from R00, recycle kept; R03 fed the %.0f%% product share '
            '(rel %.0e)' % (100 * (1 - RECYCLE), gap))


STAGES = (
    ('1 acyclic pair == SimulationExec                ',
     stage1_acyclic_matches_simexec),
    ('2 recycle, iterated in the RHS                  ',
     stage2_recycle_iterated),
    ('3 recycle torn on the other link                ',
     stage3_tear_choice_is_immaterial),
    ('4 recycle as algebraic states                   ',
     stage4_recycle_algebraic),
    ('5 failed iteration falls back to algebraic      ',
     stage5_fallback_to_algebraic),
    ('6 controller acting on a downstream unit        ',
     stage6_control_across_units),
    ('7 crystallizer pairs == SimulationExec          ',
     stage7_crystallizer_pairs),
    ('8 ConcurrentVessel as a flowsheet node          ',
     stage8_flowsheet_node),
    ('9 cascade: R02 sets the level R01 holds         ',
     stage9_cascade_control),
    ('10 crystallizer recycle carrying crystals       ',
     stage10_crystallizer_recycle),
)


def main(show_traceback=True):
    results = {}

    for label, fn in STAGES:
        try:
            detail = fn()
            results[label] = ('PASS', detail)
        except Exception as exc:
            detail = '%s: %s' % (type(exc).__name__, exc)
            results[label] = ('FAIL', detail)

            if show_traceback:
                print('\n===== stage %s traceback =====' % label.strip())
                traceback.print_exc()

    print('\n' + '=' * 78)
    print('CONCURRENT VESSEL RESULTS  (backend: %s)' % BACKEND)
    print('=' * 78)

    for label, (status, detail) in results.items():
        print('  [%s] %s %s' % (status, label, detail))

    return results


if __name__ == '__main__':
    main()
