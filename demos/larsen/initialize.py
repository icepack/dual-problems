import argparse
import subprocess
import numpy as np
import geojson
import xarray
import firedrake
from firedrake import assemble, Constant, inner, grad, dx
import icepack

parser = argparse.ArgumentParser()
parser.add_argument("--outline")
parser.add_argument("--degree", type=int, default=1)
parser.add_argument("--output", default="larsen-initial.h5")
args = parser.parse_args()

# Create the mesh and some function spaces
outline_filename = args.outline or icepack.datasets.fetch_outline("larsen-2015")
with open(outline_filename, "r") as outline_file:
    outline = geojson.load(outline_file)

geometry = icepack.meshing.collection_to_geo(outline)
with open("larsen-2015.geo", "w") as geometry_file:
    geometry_file.write(geometry.get_code())

command = "gmsh -2 -v 0 -o larsen-2015.msh larsen-2015.geo"
subprocess.run(command.split())

mesh = firedrake.Mesh("larsen-2015.msh")
Q = firedrake.FunctionSpace(mesh, "CG", args.degree)
V = firedrake.VectorFunctionSpace(mesh, "CG", args.degree)

# Load in some observational data
bedmachine_filename = icepack.datasets.fetch_bedmachine_antarctica()
bedmachine_dataset = xarray.open_dataset(bedmachine_filename)

h_obs = icepack.interpolate(bedmachine_dataset["thickness"], Q)
h = h_obs.copy(deepcopy=True)
α = Constant(2e3)
J = 0.5 * ((h - h_obs) ** 2 + α**2 * inner(grad(h), grad(h))) * dx
F = firedrake.derivative(J, h)
firedrake.solve(F == 0, h)

velocity_filename = icepack.datasets.fetch_measures_antarctica()
velocity_dataset = xarray.open_dataset(velocity_filename)
vx = velocity_dataset["VX"]
vy = velocity_dataset["VY"]
errx = velocity_dataset["ERRX"]
erry = velocity_dataset["ERRY"]

V = firedrake.VectorFunctionSpace(mesh, family="CG", degree=2)
u_obs = icepack.interpolate((vx, vy), V)
σx = icepack.interpolate(errx, Q)
σy = icepack.interpolate(erry, Q)

# Set up the model and solver
T = Constant(260)
A0 = icepack.rate_factor(T)

def viscosity(**kwargs):
    u = kwargs["velocity"]
    h = kwargs["thickness"]
    θ = kwargs["log_fluidity"]

    A = A0 * firedrake.exp(θ)
    return icepack.models.viscosity.viscosity_depth_averaged(
        velocity=u, thickness=h, fluidity=A
    )

model = icepack.models.IceShelf(viscosity=viscosity)
opts = {
    # TODO: These aren't going to be the same for every mesh so we need a way to
    # pass them as cmdline args, or just assume it's all Dirichlet
    "dirichlet_ids": [2, 4, 5, 6, 7, 8, 9],
    "diagnostic_solver_type": "petsc",
    "diagnostic_solver_parameters": {
        "snes_type": "newtontr",
        "ksp_type": "gmres",
        "pc_type": "lu",
        "pc_factor_mat_solver_type": "mumps",
    },
}
solver = icepack.solvers.FlowSolver(model, **opts)

θ = firedrake.Function(Q)
u = solver.diagnostic_solve(
    velocity=u_obs,
    thickness=h,
    log_fluidity=θ,
)

# Set up the statistical estimation problem
def simulation(θ):
    return solver.diagnostic_solve(
        velocity=u_obs,
        thickness=h,
        log_fluidity=θ,
    )

area = assemble(Constant(1.0) * dx(mesh))
Ω = Constant(area)

def loss_functional(u):
    δu = u - u_obs
    return 0.5 / Ω * ((δu[0] / σx)**2 + (δu[1] / σy)**2) * dx

def regularization(θ):
    Θ = Constant(1.)
    L = Constant(7.5e3)
    return 0.5 / Ω * (L / Θ)**2 * inner(grad(θ), grad(θ)) * dx

problem = icepack.statistics.StatisticsProblem(
    simulation=simulation,
    loss_functional=loss_functional,
    regularization=regularization,
    controls=θ,
)

estimator = icepack.statistics.MaximumProbabilityEstimator(
    problem,
    gradient_tolerance=1e-4,
    step_tolerance=1e-1,
    max_iterations=50,
)
θ = estimator.solve()
u = simulation(θ)

# Save the results to disk
with firedrake.CheckpointFile(args.output, "w") as chk:
    chk.save_function(u, name="velocity")
    chk.save_function(θ, name="log_fluidity")
