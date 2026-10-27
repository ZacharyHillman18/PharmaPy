"""Where does PharmaPy's right-hand side stop being traceable?

Feeds CasADi SX symbols and JAX tracers into each layer of a real continuous
reactor and reports the first PharmaPy line that rejects them.
"""
import os
import sys
import traceback

REPO = r"C:\Users\zhillma\OneDrive - purdue.edu\Documents\Documents\_Grad_School\mypharmadev\PharmaPy"
sys.path.insert(0, REPO)
sys.path.append(os.path.join(os.environ["TEMP"], "ppdoc", "pylib"))  # after the env's numpy

import numpy as np
import casadi as ca
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

from PharmaPy.Kinetics import RxnKinetics
from PharmaPy.Phases_Refactored import LiquidPhase
from PharmaPy.Streams_Refactored import LiquidStream
from PharmaPy.Reactors_Refactored import ContinuousReactor
from PharmaPy.ProcessControl_Refactored import (ContinuousVesselController,
                                                DefaultContinuousVesselVolume)
from PharmaPy.DataClasses import StateKey, PhaseRef

PATH = os.path.join(REPO, "tests", "Flowsheet", "data", "compound_database.json")
print("numpy", np.__version__, "| casadi", ca.__version__, "| jax", jax.__version__)


def where_it_broke(exc):
    frames = [f for f in traceback.extract_tb(exc.__traceback__)
              if os.sep + "PharmaPy" + os.sep in f.filename]
    if not frames:
        return "(outside PharmaPy)"
    f = frames[-1]
    return f"{os.path.basename(f.filename)}:{f.lineno}  `{f.line.strip()[:90]}`"


def probe(label, fn):
    try:
        out = fn()
        print(f"  OK    {label}: {type(out).__name__}")
    except Exception as exc:
        print(f"  FAIL  {label}: {type(exc).__name__}: {str(exc).splitlines()[0][:110]}")
        print(f"        at {where_it_broke(exc)}")


# ---------------------------------------------------------------------
print("\n== numpy applied to a CasADi SX / a JAX tracer (no PharmaPy) ==")
x = ca.SX.sym("x", 3)
probe("np.exp(SX)", lambda: np.exp(x))
probe("np.maximum(SX, 0)", lambda: np.maximum(x, 0.0))
probe("np.where(SX > 0, SX, 0)", lambda: np.where(x > 0, x, 0))
probe("np.sum(SX)", lambda: np.sum(x))
probe("builtin max(SX[0], 0)", lambda: max(x[0], 0.0))
probe("if SX[0] > 0", lambda: 1 if x[0] > 0 else 0)
probe("float(SX[0])", lambda: float(x[0]))
def write_into_numpy():
    buf = np.zeros(3)
    buf[0] = x[0]
    return buf
probe("numpy_buffer[0] = SX", write_into_numpy)
def write_into_sx():
    buf = ca.SX.zeros(3)
    buf[0] = x[0]
    return buf
probe("SX_buffer[0] = SX", write_into_sx)

def jax_cases():
    y = jnp.ones(3)
    def f_np(v): return np.exp(v)
    def f_inplace(v):
        v[0] = 1.0
        return v
    def f_branch(v): return v if v[0] > 0 else -v
    def f_at(v): return v.at[0].set(1.0)
    for name, f in (("np.exp(tracer)", f_np), ("tracer[0] = 1.0", f_inplace),
                    ("if tracer[0] > 0", f_branch), ("tracer.at[0].set(1)", f_at)):
        probe(name, lambda f=f: jax.make_jaxpr(f)(y))
jax_cases()

# ---------------------------------------------------------------------
print("\n== PharmaPy layers ==")
kin = RxnKinetics(path=PATH, rxn_list=['A + B --> C', 'C + A --> D'],
                  k_params=np.array([2.654e4, 5.3e2]), ea_params=np.array([4.0e4, 3.0e4]))
c_sym = ca.SX.sym("c", 3)
T_sym = ca.SX.sym("T")
probe("kinetics.get_rxn_rates(SX conc, SX temp)",
      lambda: kin.get_rxn_rates(c_sym, T_sym, return_both=True))
probe("kinetics.get_rxn_rates(jax)",
      lambda: jax.make_jaxpr(lambda c, T: kin.get_rxn_rates(c, T, return_both=True)[1])(
          jnp.array([0.3, 0.3, 0.1]), 313.15))

liq = LiquidPhase(PATH, temp=313.15, mole_conc=np.array([0.33, 0.33, 0, 0, 0]), vol=0.06,
                  name_solv='solvent')
probe("liquid.getDensity(temp=SX)", lambda: liq.getDensity(temp=T_sym))
probe("liquid.getCp(SX temp)", lambda: liq.getCp(T_sym))
m_sym = ca.SX.sym("m", 5)
probe("liquid.updatePhase(mass_j=SX)", lambda: liq.updatePhase(mass_j=m_sym))
probe("liquid.updatePhase(mass_j=jax)",
      lambda: jax.make_jaxpr(lambda m: (liq.updatePhase(mass_j=m), liq.vol)[1])(
          jnp.asarray(liq.mass_j)))

ctrl = DefaultContinuousVesselVolume()
probe("soft_floor(SX)", lambda: ctrl.soft_floor(T_sym, 1e-3))

# ---------------------------------------------------------------------
print("\n== the whole vessel: ContinuousReactor.unit_model ==")
def reactor():
    unit = ContinuousReactor(h_conv=1e4, diam=0.4,
                             controller=ContinuousVesselController(temp_func=lambda t: 313.15))
    unit.Phases = LiquidPhase(PATH, temp=313.15, mole_conc=np.array([0.33, 0.33, 0, 0, 0]),
                              vol=0.06, name_solv='solvent')
    unit.RxnKinetics = kin
    unit.Inlet = LiquidStream(PATH, temp=313.15, mole_conc=np.array([0.5, 0.5, 0, 0, 0]),
                              vol_flow=1e-5, name_solv='solvent')
    unit.compile_structure()
    return unit

unit = reactor()
y0 = unit.create_solver_init_states()
print("  states:", unit.solver_state_collection.dim, "| numeric RHS ok:",
      np.all(np.isfinite(unit.unit_model(0.0, y0))))
y_sym = ca.SX.sym("y", y0.size)
probe("unit_model(0, SX)", lambda: unit.unit_model(0.0, y_sym))
unit = reactor()
probe("unit_model(0, jax tracer)", lambda: jax.make_jaxpr(lambda y: unit.unit_model(0.0, y))(jnp.asarray(y0)))
