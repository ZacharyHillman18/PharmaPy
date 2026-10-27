"""Spike driver: trace the continuous reactor's RHS with JAX; check it matches numpy."""
import os
import sys
import traceback

SPIKE = os.path.join(os.environ["TEMP"], "ppdoc", "spike")
REPO = r"C:\Users\zhillma\OneDrive - purdue.edu\Documents\Documents\_Grad_School\mypharmadev\PharmaPy"
sys.path.insert(0, SPIKE)
sys.path.append(os.path.join(os.environ["TEMP"], "ppdoc", "pylib"))

import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

import PharmaPy
assert SPIKE in PharmaPy.__file__, PharmaPy.__file__

from PharmaPy.Kinetics import RxnKinetics
from PharmaPy.Phases_Refactored import LiquidPhase
from PharmaPy.Streams_Refactored import LiquidStream
from PharmaPy.Reactors_Refactored import ContinuousReactor
from PharmaPy.ProcessControl_Refactored import ContinuousVesselController

PATH = os.path.join(REPO, "tests", "Flowsheet", "data", "compound_database.json")


def reactor(isothermal_controller=True):
    controller = (ContinuousVesselController(temp_func=lambda t: 313.15)
                  if isothermal_controller else None)
    unit = ContinuousReactor(h_conv=1e4, diam=0.4, controller=controller)
    unit.Phases = LiquidPhase(PATH, temp=313.15, mole_conc=np.array([0.33, 0.33, 0, 0, 0]),
                              vol=0.06, name_solv='solvent')
    unit.RxnKinetics = RxnKinetics(path=PATH, rxn_list=['A + B --> C', 'C + A --> D'],
                                   k_params=np.array([2.654e4, 5.3e2]),
                                   ea_params=np.array([4.0e4, 3.0e4]))
    unit.Inlet = LiquidStream(PATH, temp=313.15, mole_conc=np.array([0.5, 0.5, 0, 0, 0]),
                              vol_flow=1e-5, name_solv='solvent')
    unit.compile_structure()
    return unit


def where_it_broke(exc):
    frames = [f for f in traceback.extract_tb(exc.__traceback__)
              if os.sep + "PharmaPy" + os.sep in f.filename]
    if not frames:
        return "(outside PharmaPy)"
    out = []
    for f in frames[-3:]:
        out.append(f"{os.path.basename(f.filename)}:{f.lineno} in {f.name}: `{f.line.strip()[:100]}`")
    return "\n        ".join(out)


case = sys.argv[1] if len(sys.argv) > 1 else "isothermal"
unit = reactor(isothermal_controller=(case == "isothermal"))
y0 = unit.create_solver_init_states()
numeric = np.array(unit.unit_model(0.0, y0), dtype=float)
print(f"[{case}] states {y0.size}; numeric RHS {numeric}")

try:
    traced_unit = reactor(isothermal_controller=(case == "isothermal"))
    traced_unit.unit_model(0.0, y0)  # numeric warm-up: lazily initialised controller state gets numbers, not tracers
    rhs = jax.jit(lambda y: traced_unit.unit_model(0.0, y))
    out = np.asarray(rhs(jnp.asarray(y0)))
    gap = np.max(np.abs(out - numeric) / np.maximum(np.abs(numeric), 1e-30))
    print(f"  TRACED. jit RHS vs numpy: max rel diff {gap:.2e}")
    y1 = y0 * (1 + 1e-3 * np.arange(1, y0.size + 1))
    ref = reactor(case == "isothermal")
    ref.unit_model(0.0, y0)                      # same warm-up as the traced copy
    n1 = np.array(ref.unit_model(0.0, y1), dtype=float)
    g1 = np.max(np.abs(np.asarray(rhs(jnp.asarray(y1))) - n1) / np.maximum(np.abs(n1), 1e-30))
    print(f"  at another state: max rel diff {g1:.2e}")
    jac = np.asarray(jax.jacfwd(rhs)(jnp.asarray(y0)))   # differentiate the compiled function
    h = 1e-6 * np.maximum(np.abs(y0), 1e-6)
    fd = np.column_stack([(np.array(ref.unit_model(0.0, y0 + h[i] * np.eye(y0.size)[i])) - numeric) / h[i]
                          for i in range(y0.size)])
    scale = np.max(np.abs(jac))
    print(f"  exact Jacobian {jac.shape}; vs finite differences: max abs diff / max |J| = {np.max(np.abs(jac - fd)) / scale:.2e}")
except Exception as exc:
    print(f"  FAIL {type(exc).__name__}: {str(exc).splitlines()[0][:160]}")
    print(f"        {where_it_broke(exc)}")
