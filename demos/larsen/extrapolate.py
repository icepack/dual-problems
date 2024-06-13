import argparse
import subprocess
import numpy as np
import geojson
import xarray
import firedrake
from firedrake import assemble, Constant, inner, grad, dx
import icepack

parser = argparse.ArgumentParser()
parser.add_argument("--input", default="larsen-initial.h5")
parser.add_argument("--outline", default="larsen.geojson")
parser.add_argument("--refinement", type=int, default=0)
parser.add_argument("--degree", type=int, default=1)
parser.add_argument("--output", default="larsen-extrapolated.h5")
args = parser.parse_args()

# Read in the input data
with firedrake.CheckpointFile(args.input, "r") as chk:
    input_mesh = chk.load_mesh()
    u_input = chk.load_function(input_mesh, name="velocity")
    θ_input = chk.load_function(input_mesh, name="log_fluidity")

    Δ = firedrake.FunctionSpace(input_mesh, "DG", 0)
    μ = firedrake.Function(Δ).interpolate(Constant(1))

# Create the mesh and some function spaces
with open(args.outline, "r") as outline_file:
    outline = geojson.load(outline_file)

geometry = icepack.meshing.collection_to_geo(outline)
with open("larsen.geo", "w") as geometry_file:
    geometry_file.write(geometry.get_code())

command = "gmsh -2 -v 0 -o larsen.msh larsen.geo"
subprocess.run(command.split())

initial_mesh = firedrake.Mesh("larsen.msh")
mesh_hierarchy = firedrake.MeshHierarchy(initial_mesh, args.refinement)
mesh = mesh_hierarchy[-1]

Q = firedrake.FunctionSpace(mesh, "CG", args.degree)
V = firedrake.VectorFunctionSpace(mesh, "CG", args.degree)

# Project the mask, log-fluidity, and velocity onto the larger mesh. The
# regions with no data will be extrapolated by zero.
Δ = firedrake.FunctionSpace(mesh, "DG", 0)
μ = firedrake.project(μ, Δ)

Eθ = firedrake.project(θ_input, Q)
Eu = firedrake.project(u_input, V)

θ = Eθ.copy(deepcopy=True)
u = Eu.copy(deepcopy=True)

α = Constant(8e3)

bc = firedrake.DirichletBC(V, Eu, [4, 7, 8, 9])
J = 0.5 * (μ * inner(u - Eu, u - Eu) + α**2 * inner(grad(u), grad(u))) * dx
F = firedrake.derivative(J, u)
firedrake.solve(F == 0, u, bc)

bc = firedrake.DirichletBC(Q, Eθ, [4, 7, 8, 9])
J = 0.5 * (μ * inner(θ - Eθ, θ - Eθ) + α**2 * inner(grad(θ), grad(θ))) * dx
F = firedrake.derivative(J, θ)
firedrake.solve(F == 0, θ, bc)

with firedrake.CheckpointFile(args.output, "w") as chk:
    chk.save_mesh(mesh)
    chk.save_function(u, name="velocity")
    chk.save_function(θ, name="log_fluidity")
    chk.save_function(μ, name="mask")
