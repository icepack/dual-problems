import argparse
import subprocess
import numpy as np
import geojson
import xarray
import firedrake
from firedrake import assemble, Constant, inner, grad, dx, ds, dS
import icepack
from icepack.constants import glen_flow_law as n
from dualform import ice_shelf

parser = argparse.ArgumentParser()
parser.add_argument("--input", default="larsen-extrapolated.h5")
parser.add_argument("--timesteps-per-year", type=int, default=12)
parser.add_argument("--final-time", type=float, default=8.0)
parser.add_argument("--degree", type=int, default=1)
parser.add_argument("--output", default="larsen-simulation.h5")
args = parser.parse_args()

# Read in the starting data
with firedrake.CheckpointFile(args.input, "r") as chk:
    mesh = chk.load_mesh()
    u = chk.load_function(mesh, name="velocity")
    θ = chk.load_function(mesh, name="log_fluidity")
    μ_obs = chk.load_function(mesh, name="mask")

Q = firedrake.FunctionSpace(mesh, "CG", args.degree)
Δ = firedrake.FunctionSpace(mesh, "DG", args.degree)
V = firedrake.VectorFunctionSpace(mesh, "CG", args.degree)
Σ = firedrake.TensorFunctionSpace(mesh, "DG", args.degree - 1, symmetry=True)
Z = V * Σ

u_in = firedrake.project(u, V)
θ = firedrake.project(θ, Q)

z = firedrake.Function(Z)
z.sub(0).assign(u_in)

# Read in the thickness map
bedmachine_filename = icepack.datasets.fetch_bedmachine_antarctica()
bedmachine_dataset = xarray.open_dataset(bedmachine_filename)

h_obs = icepack.interpolate(bedmachine_dataset["thickness"], Q)
h = h_obs.copy(deepcopy=True)
α = Constant(2e3)
J = 0.5 * ((h - h_obs) ** 2 + α**2 * inner(grad(h), grad(h))) * dx
F = firedrake.derivative(J, h)
firedrake.solve(F == 0, h)
h = firedrake.interpolate(h, Δ)

μ = firedrake.project(μ_obs, Q)
J = 0.5 * ((μ - μ_obs) ** 2 + α**2 * inner(grad(μ), grad(μ))) * dx
F = firedrake.derivative(J, μ)
firedrake.solve(F == 0, μ)

μ_cutoff = Constant(0.05)
h.interpolate(firedrake.conditional(μ < μ_cutoff, 0, h))

# Set up the momentum balance equation and solver
T = Constant(260)
A0 = icepack.rate_factor(T)

ε_c = firedrake.Constant(0.01)
A = A0 * firedrake.exp(θ)
τ_c = firedrake.interpolate((ε_c / A)**(1 / n), Q)

fns = [
    ice_shelf.viscous_power,
    ice_shelf.boundary,
    ice_shelf.constraint,
    ice_shelf.constraint_edges,
]

u, M = firedrake.split(z)
fields = {
    "velocity": u,
    "membrane_stress": M,
    "thickness": h,
}

h_min = Constant(10.0)
rfields = {
    "velocity": u,
    "membrane_stress": M,
    "thickness": firedrake.max_value(h_min, h),
}

params = {
    "viscous_yield_strain": ε_c,
    "viscous_yield_stress": τ_c,
    "outflow_ids": (1, 2),
}

L_1 = sum(fn(**rfields, **params, flow_law_exponent=1) for fn in fns)
F_1 = firedrake.derivative(L_1, z)

L = sum(fn(**fields, **params, flow_law_exponent=n) for fn in fns)
F = firedrake.derivative(L, z)

L_r = sum(fn(**rfields, **params, flow_law_exponent=n) for fn in fns)
F_r = firedrake.derivative(L_r, z)
J_r = firedrake.derivative(F_r, z)

qdegree = int(n) + 2
inflow_ids = (4, 7, 8, 9)
ice_rise_ids = (6,)
bc_inflow = firedrake.DirichletBC(Z.sub(0), u_in, inflow_ids)
bc_ice_rise = firedrake.DirichletBC(Z.sub(0), 0, ice_rise_ids)
bcs = [bc_inflow, bc_ice_rise]
problem_params = {
    "form_compiler_parameters": {"quadrature_degree": qdegree}, "bcs": bcs
}
solver_params = {
    "solver_parameters": {
        "snes_monitor": None,
        "snes_max_it": 10,
        "snes_convergence_test": "skip",
        "snes_type": "newtonls",
        "ksp_type": "gmres",
        "pc_type": "lu",
        "pc_factor_mat_solver_type": "mumps",
    },
}
firedrake.solve(F_1 == 0, z, **problem_params, **solver_params)

u_problem = firedrake.NonlinearVariationalProblem(F, z, J=J_r, **problem_params)
u_solver = firedrake.NonlinearVariationalSolver(u_problem, **solver_params)
u_solver.solve()

# Set up the mass balance equation
h_n = h.copy(deepcopy=True)
h0 = h.copy(deepcopy=True)
φ = firedrake.TestFunction(h.function_space())
dt = Constant(1.0 / args.timesteps_per_year)
flux_cells = ((h - h_n) / dt * φ - inner(h * u, grad(φ))) * dx
ν = firedrake.FacetNormal(mesh)
f = h * firedrake.max_value(0, inner(u, ν))
flux_facets = (f("+") - f("-")) * (φ("+") - φ("-")) * dS
flux_in = h0 * firedrake.min_value(0, inner(u, ν)) * φ * ds
flux_out = h * firedrake.max_value(0, inner(u, ν)) * φ * ds
G = flux_cells + flux_facets + flux_in + flux_out
h_problem = firedrake.NonlinearVariationalProblem(G, h)
h_solver = firedrake.NonlinearVariationalSolver(h_problem)

# Load in the outline of Larsen in 2019
outline_filename = icepack.datasets.fetch_outline("larsen-2019")
with open(outline_filename, "r") as outline_file:
    outline = geojson.load(outline_file)

geometry = icepack.meshing.collection_to_geo(outline)
with open("larsen-2019.geo", "w") as geometry_file:
    geometry_file.write(geometry.get_code())

command = "gmsh -2 -v 0 -o larsen-2019.msh larsen-2019.geo"
subprocess.run(command.split())

# Create a mask to remove ice where the calving event occurred
mesh_2019 = firedrake.Mesh("larsen-2019.msh")
Q_2019 = firedrake.FunctionSpace(mesh_2019, "CG", 1)
μ = firedrake.Function(Q_2019)
μ.assign(Constant(1.0))
μ = firedrake.project(μ, Q)
μ.interpolate(firedrake.max_value(0, firedrake.min_value(1, μ)))

h_c = firedrake.Constant(1.0)
with firedrake.CheckpointFile(args.output, "w") as chk:
    chk.save_function(h, name="thickness", idx=0)

    time_to_calve = 4.0
    num_steps = int(args.final_time * args.timesteps_per_year) + 1
    timesteps = np.linspace(0.0, args.final_time, num_steps)
    for step, t in enumerate(timesteps):
        if abs(t - time_to_calve) < float(dt) / 2:
            print("IT CALVING NOW")
            h.interpolate(μ * h)
            h_n.assign(h)
            u_solver.solve()

        h_solver.solve()
        h.interpolate(firedrake.conditional(h < h_c, 0, h))
        h_n.assign(h)
        u_solver.solve()
        chk.save_function(h, name="thickness", idx=step + 1)

    chk.h5pyfile.create_dataset("timesteps", data=timesteps)
