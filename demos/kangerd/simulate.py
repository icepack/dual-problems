import argparse
import subprocess
import tqdm
import numpy as np
import xarray
import firedrake
from firedrake import assemble, Constant, exp, max_value, inner, grad, dx, ds, dS
import icepack
from icepack2.constants import (
    glen_flow_law as n,
    weertman_sliding_law as m,
    ice_density as ρ_I,
    water_density as ρ_W,
)
from icepack2 import model

parser = argparse.ArgumentParser()
parser.add_argument("--input", default="kangerdlugssuaq-extrapolated.h5")
parser.add_argument("--timesteps-per-year", type=int, default=96)
parser.add_argument("--final-time", type=float, default=0.5)
parser.add_argument("--degree", type=int, default=1)
parser.add_argument("--output", default="kangerdlugssuaq-simulation.h5")
args = parser.parse_args()

# Read in the starting data
with firedrake.CheckpointFile(args.input, "r") as chk:
    mesh = chk.load_mesh()
    q = chk.load_function(mesh, name="log_friction")
    τ_c = chk.h5pyfile.attrs["mean_stress"]
    u_c = chk.h5pyfile.attrs["mean_speed"]

    timesteps = np.array(chk.h5pyfile["timesteps"])
    u = chk.load_function(mesh, name="velocity", idx=len(timesteps) - 1)
    h = chk.load_function(mesh, name="thickness", idx=len(timesteps) - 1)

Q = firedrake.FunctionSpace(mesh, "CG", args.degree)
Δ = firedrake.FunctionSpace(mesh, "DG", args.degree)
V = firedrake.VectorFunctionSpace(mesh, "CG", args.degree)
Σ = firedrake.TensorFunctionSpace(mesh, "DG", args.degree - 1, symmetry=True)
T = firedrake.VectorFunctionSpace(mesh, "DG", args.degree - 1)
Z = V * Σ * T

u_in = firedrake.project(u, V)
q = firedrake.project(q, Q)

z = firedrake.Function(Z)
z.sub(0).assign(u_in)

# Read the thickness and bed data
bedmachine = xarray.open_dataset(icepack.datasets.fetch_bedmachine_greenland())
b = icepack.interpolate(bedmachine["bed"], Q)
h = firedrake.project(h, Δ)
s = firedrake.project(max_value(b + h, (1 - ρ_I / ρ_W) * h), Δ)

# Set up the momentum balance equation and solve
A = icepack.rate_factor(Constant(260))
ε_c = Constant(A * τ_c ** n)
print(f"τ_c: {1000 * float(τ_c):.1f} kPa")
print(f"ε_c: {1000 * float(ε_c):.1f} (m / yr) / km")
print(f"u_c: {float(u_c):.1f} m / yr")

fns = [
    model.viscous_power,
    model.friction_power,
    #model.calving_terminus,
    model.momentum_balance,
]

u, M, τ = firedrake.split(z)
fields = {
    "velocity": u,
    "membrane_stress": M,
    "basal_stress": τ,
    "thickness": h,
    "surface": s,
}

h_min = Constant(10.0)
rfields = {
    "velocity": u,
    "membrane_stress": M,
    "basal_stress": τ,
    "thickness": max_value(h_min, h),
    "surface": s,
}

rheology = {
    "flow_law_exponent": n,
    "flow_law_coefficient": ε_c / τ_c**n,
    "sliding_exponent": m,
    "sliding_coefficient": u_c / τ_c**m * exp(m * q),
}

linear_rheology = {
    "flow_law_exponent": 1,
    "flow_law_coefficient": ε_c / τ_c,
    "sliding_exponent": 1,
    "sliding_coefficient": u_c / τ_c * exp(q),
}

L_1 = sum(fn(**rfields, **linear_rheology) for fn in fns)
F_1 = firedrake.derivative(L_1, z)
J_1 = firedrake.derivative(F_1, z)

L = sum(fn(**fields, **rheology) for fn in fns)
F = firedrake.derivative(L, z)

L_r = sum(fn(**rfields, **rheology) for fn in fns)
F_r = firedrake.derivative(L_r, z)
J_r = firedrake.derivative(F_r, z)
α = firedrake.Constant(0.0)
J = J_r + α * J_1

inflow_ids = [1]
bc_in = firedrake.DirichletBC(Z.sub(0), u_in, inflow_ids)
outflow_ids = [2, 3, 4]
bc_out = firedrake.DirichletBC(Z.sub(0), Constant((0.0, 0.0)), outflow_ids)
bcs = [bc_in, bc_out]

qdegree = int(max(m, n)) + 2
problem_params = {
    "form_compiler_parameters": {"quadrature_degree": qdegree},
    "bcs": bcs,
}
solver_params = {
    "solver_parameters": {
        #"snes_monitor": None,
        #"snes_converged_reason": None,
        "snes_stol": 0.0,
        "snes_rtol": 1e-6,
        "snes_divergence_tolerance": -1,
        "snes_max_it": 200,
        "snes_type": "newtonls",
        "ksp_type": "gmres",
        "pc_type": "lu",
        "pc_factor_mat_solver_type": "mumps",
    },
}
firedrake.solve(F_1 == 0, z, **problem_params, **solver_params)

u_problem = firedrake.NonlinearVariationalProblem(F, z, J=J, **problem_params)
u_solver = firedrake.NonlinearVariationalSolver(u_problem, **solver_params)
u_solver.solve()

# Set up the mass balance equation
h_n = h.copy(deepcopy=True)
h0 = h.copy(deepcopy=True)
φ = firedrake.TestFunction(h.function_space())
dt = Constant(1.0 / args.timesteps_per_year)
flux_cells = ((h - h_n) / dt * φ - inner(h * u, grad(φ))) * dx
ν = firedrake.FacetNormal(mesh)
f = h * max_value(0, inner(u, ν))
flux_facets = (f("+") - f("-")) * (φ("+") - φ("-")) * dS
flux_in = h0 * firedrake.min_value(0, inner(u, ν)) * φ * ds
flux_out = h * max_value(0, inner(u, ν)) * φ * ds
G = flux_cells + flux_facets + flux_in + flux_out
h_problem = firedrake.NonlinearVariationalProblem(G, h)
h_solver = firedrake.NonlinearVariationalSolver(h_problem)

# Run the simulation
h_c = Constant(5.0)
num_steps = int(args.final_time * args.timesteps_per_year) + 1
with firedrake.CheckpointFile(args.output, "w") as chk:
    u, M, τ = z.subfunctions
    chk.save_function(h, name="thickness", idx=0)
    chk.save_function(u, name="velocity", idx=0)
    chk.save_function(M, name="membrane_stress", idx=0)
    chk.save_function(τ, name="basal_stress", idx=0)

    timesteps = np.linspace(0.0, args.final_time, num_steps)
    for step in tqdm.trange(num_steps):
        h_solver.solve()
        h.interpolate(firedrake.conditional(h < h_c, 0, h))
        h_n.assign(h)
        s.interpolate(max_value(b + h, (1 - ρ_I / ρ_W) * h))
        u_solver.solve()

        # Save the results to disk
        u, M, τ = z.subfunctions
        chk.save_function(h, name="thickness", idx=step + 1)
        chk.save_function(u, name="velocity", idx=step + 1)
        chk.save_function(M, name="membrane_stress", idx=step + 1)
        chk.save_function(τ, name="basal_stress", idx=step + 1)

    chk.save_function(q, name="log_friction")
    chk.h5pyfile.attrs["mean_stress"] = τ_c
    chk.h5pyfile.attrs["mean_speed"] = u_c
    chk.h5pyfile.create_dataset("timesteps", data=timesteps)
