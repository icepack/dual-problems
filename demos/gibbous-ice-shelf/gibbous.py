import argparse
import subprocess
import numpy as np
from numpy import pi as π
import tqdm
import pygmsh
import firedrake
from firedrake import (
    inner,
    grad,
    dx,
    ds,
    NonlinearVariationalProblem,
    NonlinearVariationalSolver,
)
from icepack.constants import glen_flow_law
from dualform import ice_shelf

parser = argparse.ArgumentParser()
parser.add_argument("--output", default="gibbous.h5")
args = parser.parse_args()

# Generate and load the mesh
R = 200e3
δx = 5e3

geometry = pygmsh.built_in.Geometry()

x1 = geometry.add_point([-R, 0, 0], lcar=δx)
x2 = geometry.add_point([+R, 0, 0], lcar=δx)

center1 = geometry.add_point([0, 0, 0], lcar=δx)
center2 = geometry.add_point([0, -4 * R, 0], lcar=δx)

arcs = [
    geometry.add_circle_arc(x1, center1, x2),
    geometry.add_circle_arc(x2, center2, x1),
]

line_loop = geometry.add_line_loop(arcs)
plane_surface = geometry.add_plane_surface(line_loop)

physical_lines = [geometry.add_physical(arc) for arc in arcs]
physical_surface = geometry.add_physical(plane_surface)

with open("ice-shelf.geo", "w") as geo_file:
    geo_file.write(geometry.get_code())

command = "gmsh -2 -format msh2 -v 0 -o ice-shelf.msh ice-shelf.geo"
subprocess.run(command.split())

mesh = firedrake.Mesh("ice-shelf.msh")

# Generate the initial data
inlet_angles = π * np.array([-3 / 4, -1 / 2, -1 / 3, -1 / 6])
inlet_widths = π * np.array([1 / 8, 1 / 12, 1 / 24, 1 / 12])

x = firedrake.SpatialCoordinate(mesh)

u_in = 300
h_in = 350
hb = 100
dh, du = 400, 250

hs, us = [], []
for θ, ϕ in zip(inlet_angles, inlet_widths):
    x0 = R * firedrake.as_vector((np.cos(θ), np.sin(θ)))
    v = -firedrake.as_vector((np.cos(θ), np.sin(θ)))
    L = inner(x - x0, v)
    W = x - x0 - L * v
    Rn = 2 * ϕ / π * R
    q = firedrake.max_value(1 - (W / Rn) ** 2, 0)
    hs.append(hb + q * ((h_in - hb) - dh * L / R))
    us.append(firedrake.exp(-4 * (W / R) ** 2) * (u_in + du * L / R) * v)

h_expr = firedrake.Constant(hb)
for h in hs:
    h_expr = firedrake.max_value(h, h_expr)

u_expr = sum(us)

# Create some function spaces
cg = firedrake.FiniteElement("CG", "triangle", 1)
dg = firedrake.FiniteElement("DG", "triangle", 0)
Q = firedrake.FunctionSpace(mesh, cg)
V = firedrake.VectorFunctionSpace(mesh, cg)
Σ = firedrake.TensorFunctionSpace(mesh, dg, symmetry=True)
Z = V * Σ

h0 = firedrake.interpolate(h_expr, Q)
u0 = firedrake.interpolate(u_expr, V)

ε_c = firedrake.Constant(0.01)
τ_c = firedrake.Constant(0.1)

# Set up the diagnostic problem and compute an initial guess by solving a
# Picard linearization of the problem
h = h0.copy(deepcopy=True)
z = firedrake.Function(Z)
u, M = firedrake.split(z)
z.sub(0).assign(u0)
kwargs = {
    "velocity": u,
    "membrane_stress": M,
    "thickness": h,
    "viscous_yield_strain": ε_c,
    "viscous_yield_stress": τ_c,
    "outflow_ids": (2,),
}

fns = [ice_shelf.viscous_power, ice_shelf.boundary, ice_shelf.constraint]
J_l = sum(fn(**kwargs, flow_law_exponent=1) for fn in fns)
F_l = firedrake.derivative(J_l, z)

J = sum(fn(**kwargs, flow_law_exponent=glen_flow_law) for fn in fns)
F = firedrake.derivative(J, z)

qdegree = int(glen_flow_law) + 2
bc = firedrake.DirichletBC(Z.sub(0), u0, (1,))
pparams = {
    #"form_compiler_parameters": {"quadrature_degree": qdegree},
    "bcs": bc,
}
sparams = {
    "solver_parameters": {
        "snes_type": "newtonls",
        "ksp_type": "gmres",
        "pc_type": "lu",
        "pc_factor_mat_solver_type": "mumps",
    },
}
firedrake.solve(F_l == 0, z, **pparams, **sparams)

# Set up the diagnostic problem and solver
diagnostic_problem = NonlinearVariationalProblem(F, z, **pparams)
diagnostic_solver = NonlinearVariationalSolver(diagnostic_problem, **sparams)
diagnostic_solver.solve()

# Set up the prognostic problem and solver
h_n = h.copy(deepcopy=True)
φ = firedrake.TestFunction(Q)
final_time = 400.0
num_steps = 200
dt = firedrake.Constant(final_time / num_steps)
flux_cells = ((h - h_n) / dt * φ - inner(h * u, grad(φ))) * dx
ν = firedrake.FacetNormal(mesh)
flux_in = h0 * firedrake.min_value(0, inner(u, ν)) * φ * ds
flux_out = h * firedrake.max_value(0, inner(u, ν)) * φ * ds
G = flux_cells + flux_in + flux_out
prognostic_problem = NonlinearVariationalProblem(G, h)
prognostic_solver = NonlinearVariationalSolver(prognostic_problem)

for step in tqdm.trange(num_steps):
    prognostic_solver.solve()
    h_n.assign(h)
    diagnostic_solver.solve()

u, M = z.subfunctions
with firedrake.CheckpointFile(args.output, "w") as chk:
    chk.save_function(u, name="velocity")
    chk.save_function(M, name="membrane_stress")
    chk.save_function(h, name="thickness")
