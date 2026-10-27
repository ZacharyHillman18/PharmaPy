"""JAX-traceable PharmaPy -> CasADi, without a CasADi port of the code.

The reactor RHS is compiled by JAX; a casadi.Callback evaluates it, and a
second Callback gives CasADi the exact Jacobian from jax.jacfwd. CasADi's
own CVODES then integrates the reactor; the result is compared with a scipy
solve of the same unit.
"""
import os
import sys
import time
import warnings

warnings.simplefilter("ignore")
SPIKE = os.path.join(os.environ["TEMP"], "ppdoc", "spike")
sys.path.insert(0, SPIKE)
sys.path.append(os.path.join(os.environ["TEMP"], "ppdoc", "pylib"))

import numpy as np
import casadi as ca
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import runpy

g = runpy.run_path(os.path.join(SPIKE, "driver.py"), run_name="not_main")
reactor, y0 = g["reactor"], g["y0"]
n = y0.size

traced = reactor(True)
traced.unit_model(0.0, y0)                                   # numeric warm-up
rhs = jax.jit(lambda y: traced.unit_model(0.0, y))
jac = jax.jit(jax.jacfwd(rhs))   # derived from the one trace, not a second pass through the objects


class Jacobian(ca.Callback):
    def __init__(self):
        ca.Callback.__init__(self)
        self.construct("jac_rhs", {})

    def get_n_in(self): return 2                             # x, and the nominal output
    def get_n_out(self): return 1
    def get_sparsity_in(self, i): return ca.Sparsity.dense(n, 1)
    def get_sparsity_out(self, i): return ca.Sparsity.dense(n, n)

    def eval(self, arg):
        return [np.asarray(jac(jnp.asarray(np.ravel(arg[0].full()))))]


class RHS(ca.Callback):
    def __init__(self):
        ca.Callback.__init__(self)
        self.construct("rhs", {})

    def get_n_in(self): return 1
    def get_n_out(self): return 1
    def get_sparsity_in(self, i): return ca.Sparsity.dense(n, 1)
    def get_sparsity_out(self, i): return ca.Sparsity.dense(n, 1)

    def eval(self, arg):
        return [np.asarray(rhs(jnp.asarray(np.ravel(arg[0].full()))))]

    def has_jacobian(self): return True

    def get_jacobian(self, name, inames, onames, opts):
        self._jac = Jacobian()                               # keep a reference alive
        return self._jac


f = RHS()
x = ca.MX.sym("x", n)
grid = np.linspace(0.0, 3600.0, 13)
t0 = time.perf_counter()
F = ca.integrator("F", "cvodes", {"x": x, "ode": f(x)}, 0.0, grid[1:],
                  {"abstol": 1e-10, "reltol": 1e-10})
casadi_states = np.asarray(F(x0=y0)["xf"].full()).T
t_casadi = time.perf_counter() - t0

ref = reactor(True)
ref.integrator.options.update({"atol": 1e-10, "rtol": 1e-10})
t_ref, y_ref = ref.solve_unit(time_grid=grid, verbose=False)
scipy_states = np.asarray(y_ref)[1:]

gap = np.max(np.abs(casadi_states - scipy_states) / np.maximum(np.abs(scipy_states), 1e-12))
print(f"BRIDGE: CasADi CVODES on the JAX-compiled RHS, 1 h, {n} states, {t_casadi:.1f} s")
print(f"BRIDGE: vs scipy solve of the same reactor: max rel diff {gap:.2e}")
print("BRIDGE: final masses casadi", np.round(casadi_states[-1], 6))
print("BRIDGE: final masses scipy ", np.round(scipy_states[-1], 6))
