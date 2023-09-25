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
parser.add_argument("--input")
parser.add_argument("--resolution", type=float, default=5e3)
parser.add_argument("--final-time", type=float, default=400.0)
parser.add_argument("--num-steps", type=int, default=200)
parser.add_argument("--calving-freq", type=float, default=0.0)
parser.add_argument("--output", default="steady-state.h5")
args = parser.parse_args()

# Generate and load the mesh
R = 200e3
δx = args.resolution

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

# Create some function spaces and some fields
cg = firedrake.FiniteElement("CG", "triangle", 1)
dg = firedrake.FiniteElement("DG", "triangle", 0)
Q = firedrake.FunctionSpace(mesh, cg)
V = firedrake.VectorFunctionSpace(mesh, cg)
Σ = firedrake.TensorFunctionSpace(mesh, dg, symmetry=True)
Z = V * Σ

h0 = firedrake.interpolate(h_expr, Q)
u0 = firedrake.interpolate(u_expr, V)

h = h0.copy(deepcopy=True)
z = firedrake.Function(Z)

# Create some physical constants. Rather than specify the fluidity factor `A`
# in Glen's flow law, we instead define it in terms of a stress scale `τ_c` and
# a strain rate scale `ε_c` as
#
#     A = ε_c / τ_c ** n
#
# where `n` is the Glen flow law exponent. This way, we can do a continuation-
# type method in `n` while preserving dimensional correctness.
ε_c = firedrake.Constant(0.01)
τ_c = firedrake.Constant(0.1)

# If the we passed in some input data, project the inputs into our fn spaces
if args.input:
    with firedrake.CheckpointFile(args.input, "r") as chk:
        input_mesh = chk.load_mesh()
        idx = len(chk.h5pyfile["timesteps"]) - 1
        u_input = chk.load_function(input_mesh, "velocity", idx=idx)
        M_input = chk.load_function(input_mesh, "membrane_stress", idx=idx)
        h_input = chk.load_function(input_mesh, "thickness", idx=idx)

    u_projected = firedrake.project(u_input, V)
    #M_projected = firedrake.project(M_input, Σ)
    h_projected = firedrake.project(h_input, Q)

    z.sub(0).assign(u_projected)
    #z.sub(1).assign(M_projected)
    h0.assign(h_projected)
    h.assign(h_projected)
else:
    z.sub(0).assign(u0)

# Set up the diagnostic problem and compute an initial guess by solving a
# Picard linearization of the problem
u, M = firedrake.split(z)
fields = {
    "velocity": u,
    "membrane_stress": M,
    "thickness": h,
}

h_min = firedrake.Constant(1.0)
rfields = {
    "velocity": u,
    "membrane_stress": M,
    "thickness": firedrake.max_value(h_min, h),
}

params = {
    "viscous_yield_strain": ε_c,
    "viscous_yield_stress": τ_c,
    "outflow_ids": (2,),
}

fns = [ice_shelf.viscous_power, ice_shelf.boundary, ice_shelf.constraint]
L_1 = sum(fn(**fields, **params, flow_law_exponent=1) for fn in fns)
F_1 = firedrake.derivative(L_1, z)

L = sum(fn(**fields, **params, flow_law_exponent=glen_flow_law) for fn in fns)
F = firedrake.derivative(L, z)

L_r = sum(fn(**rfields, **params, flow_law_exponent=glen_flow_law) for fn in fns)
F_r = firedrake.derivative(L_r, z)
J_r = firedrake.derivative(F_r, z)

qdegree = int(glen_flow_law) + 2
bc = firedrake.DirichletBC(Z.sub(0), u0, (1,))
problem_params = {
    #"form_compiler_parameters": {"quadrature_degree": qdegree},
    "bcs": bc,
}
solver_params = {
    "solver_parameters": {
        "snes_type": "newtonls",
        "ksp_type": "gmres",
        "pc_type": "lu",
        "pc_factor_mat_solver_type": "mumps",
        "snes_rtol": 1e-2,
    },
}
firedrake.solve(F_1 == 0, z, **problem_params, **solver_params)

# Create a regularized Lagrangian

# Set up the diagnostic problem and solver
# TODO: possibly use a regularized Jacobian
diagnostic_problem = NonlinearVariationalProblem(F, z, J=J_r, **problem_params)
diagnostic_solver = NonlinearVariationalSolver(diagnostic_problem, **solver_params)
diagnostic_solver.solve()

# Set up the prognostic problem and solver
h_n = h.copy(deepcopy=True)
φ = firedrake.TestFunction(Q)
dt = firedrake.Constant(args.final_time / args.num_steps)
flux_cells = ((h - h_n) / dt * φ - inner(h * u, grad(φ))) * dx
ν = firedrake.FacetNormal(mesh)
flux_in = h0 * firedrake.min_value(0, inner(u, ν)) * φ * ds
flux_out = h * firedrake.max_value(0, inner(u, ν)) * φ * ds
G = flux_cells + flux_in + flux_out
prognostic_problem = NonlinearVariationalProblem(G, h)
prognostic_solver = NonlinearVariationalSolver(prognostic_problem)

# Set up a calving mask
R = firedrake.Constant(60e3)
y = firedrake.Constant((0.0, R))
mask = firedrake.conditional(inner(x - y, x - y) < R**2, 0.0, 1.0)

time_since_calving = 0.0

# Run the simulation and write the complete output to disk
with firedrake.CheckpointFile(args.output, "w") as chk:
    u, M = z.subfunctions
    chk.save_function(u, name="velocity", idx=0)
    chk.save_function(M, name="membrane_stress", idx=0)
    chk.save_function(h, name="thickness", idx=0)

    for step in tqdm.trange(args.num_steps):
        prognostic_solver.solve()

        if args.calving_freq != 0.0:
            if time_since_calving > args.calving_freq:
                time_since_calving = 0.0
                h.interpolate(mask * h)
            time_since_calving += float(dt)

        h.interpolate(firedrake.max_value(0, h))
        h_n.assign(h)

        diagnostic_solver.solve()

        u, M = z.subfunctions
        chk.save_function(u, name="velocity", idx=step + 1)
        chk.save_function(M, name="membrane_stress", idx=step + 1)
        chk.save_function(h, name="thickness", idx=step + 1)

    timesteps = np.linspace(0.0, args.final_time, args.num_steps + 1)
    chk.h5pyfile.create_dataset("timesteps", data=timesteps)
