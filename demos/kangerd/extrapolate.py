import argparse
import subprocess
import numpy as np
import geojson
import xarray
import firedrake
from firedrake import assemble, Constant, inner, grad, dx
import icepack

parser = argparse.ArgumentParser()
parser.add_argument("--input", default="kangerdlugssuaq-initial.h5")
parser.add_argument("--outline", default="kangerdlugssuaq-enlarged.geojson")
parser.add_argument("--refinement", type=int, default=0)
parser.add_argument("--degree", type=int, default=1)
parser.add_argument("--output", default="kangerdlugssuaq-extrapolated.h5")
args = parser.parse_args()

# Read in the input data
with firedrake.CheckpointFile(args.input, "r") as chk:
    input_mesh = chk.load_mesh()
    u_input = chk.load_function(input_mesh, name="velocity")
    q_input = chk.load_function(input_mesh, name="log_friction")
    τ_avg = chk.h5pyfile.attrs["mean_stress"]
    u_avg = chk.h5pyfile.attrs["mean_speed"]

Δ = firedrake.FunctionSpace(input_mesh, "DG", 0)
μ = firedrake.interpolate(Constant(1), Δ)

# Create the mesh and some function spaces
with open(args.outline, "r") as outline_file:
    outline = geojson.load(outline_file)

geometry = icepack.meshing.collection_to_geo(outline)
with open("kangerdlugssuaq-enlarged.geo", "w") as geometry_file:
    geometry_file.write(geometry.get_code())

command = "gmsh -2 -v 0 -o kangerdlugssuaq-enlarged.msh kangerdlugssuaq-enlarged.geo"
subprocess.run(command.split())

initial_mesh = firedrake.Mesh("kangerdlugssuaq-enlarged.msh")
mesh_hierarchy = firedrake.MeshHierarchy(initial_mesh, args.refinement)
mesh = mesh_hierarchy[-1]

Q = firedrake.FunctionSpace(mesh, "CG", args.degree)
V = firedrake.VectorFunctionSpace(mesh, "CG", args.degree)

# Project the mask, log-fluidity, and velocity onto the larger mesh. The
# regions with no data will be extrapolated by zero.
Δ = firedrake.FunctionSpace(mesh, "DG", 0)
μ = firedrake.project(μ, Δ)

Eq = firedrake.project(q_input, Q)
Eu = firedrake.project(u_input, V)

q = Eq.copy(deepcopy=True)
u = Eu.copy(deepcopy=True)

# TODO: adjust this
α = Constant(5e2)

bc_ids = [1, 2, 4]
bc = firedrake.DirichletBC(V, Eu, bc_ids)
J = 0.5 * (μ * inner(u - Eu, u - Eu) + α**2 * inner(grad(u), grad(u))) * dx
F = firedrake.derivative(J, u)
firedrake.solve(F == 0, u, bc)

bc = firedrake.DirichletBC(Q, Eq, bc_ids)
J = 0.5 * (μ * inner(q - Eq, q - Eq) + α**2 * inner(grad(q), grad(q))) * dx
F = firedrake.derivative(J, q)
firedrake.solve(F == 0, q, bc)

# Read in the thickness and surface data and project them onto the larger mesh
bedmachine = xarray.open_dataset(icepack.datasets.fetch_bedmachine_greenland())
h_obs = icepack.interpolate(bedmachine["thickness"], Q)

h = h_obs.copy(deepcopy=True)
λ = Constant(2e3)
J = 0.5 * ((h - h_obs)**2 + λ**2 * inner(grad(h), grad(h))) * dx
F = firedrake.derivative(J, h)
firedrake.solve(F == 0, h)

# Write the results to disk
with firedrake.CheckpointFile(args.output, "w") as chk:
    chk.save_mesh(mesh)
    chk.save_function(μ, name="mask")
    chk.save_function(q, name="log_friction")
    chk.save_function(u, name="velocity", idx=0)
    chk.save_function(h, name="thickness", idx=0)
    chk.h5pyfile.attrs["mean_stress"] = τ_avg
    chk.h5pyfile.attrs["mean_speed"] = u_avg
    chk.h5pyfile.create_dataset("timesteps", data=np.array([0]))
