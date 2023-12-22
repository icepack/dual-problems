import sys
import argparse
import tqdm
import numpy as np
import firedrake
from firedrake import (
    sqrt,
    exp,
    min_value,
    max_value,
    as_vector,
    Constant,
    interpolate,
    inner,
    grad,
    dx,
    ds,
    dS,
    NonlinearVariationalProblem,
    NonlinearVariationalSolver,
)
import irksome
from irksome import Dt
from icepack.constants import (
    ice_density as ρ_I,
    water_density as ρ_W,
    gravity as g,
    glen_flow_law as n,
    weertman_sliding_law as m,
)
from dualform import ice_stream


# Get the command-line options.
parser = argparse.ArgumentParser()
parser.add_argument("--num-cells", type=int, default=20)
parser.add_argument("--final-time", type=float, default=2000.0)
parser.add_argument("--timestep", type=float, default=2.5)
parser.add_argument("--output", default="mismip.h5")
args = parser.parse_args()

# Set up the mesh and some function spaces. TODO: Make it go for higher degree.
lx, ly = 640e3, 80e3
Lx, Ly = Constant(lx), Constant(ly)
ny = args.num_cells
nx = int(lx / ly) * ny
area = lx * ly

mesh = firedrake.RectangleMesh(nx, ny, lx, ly, diagonal="crossed")

cg1 = firedrake.FiniteElement("CG", "triangle", 1)
dg0 = firedrake.FiniteElement("DG", "triangle", 0)
dg1 = firedrake.FiniteElement("DG", "triangle", 1)

S = firedrake.FunctionSpace(mesh, cg1)
Q = firedrake.FunctionSpace(mesh, dg1)
V = firedrake.VectorFunctionSpace(mesh, cg1)
Σ = firedrake.TensorFunctionSpace(mesh, dg0, symmetry=True)
T = firedrake.VectorFunctionSpace(mesh, dg0)
Z = V * Σ * T

# Set up the basal elevation.
x, y = firedrake.SpatialCoordinate(mesh)

x_c = Constant(300e3)
X = x / x_c

B_0 = Constant(-150)
B_2 = Constant(-728.8)
B_4 = Constant(343.91)
B_6 = Constant(-50.57)
B_x = B_0 + B_2 * X**2 + B_4 * X**4 + B_6 * X**6

f_c = Constant(4e3)
d_c = Constant(500)
w_c = Constant(24e3)

B_y = d_c * (
    1 / (1 + exp(-2 * (y - Ly / 2 - w_c) / f_c)) +
    1 / (1 + exp(+2 * (y - Ly / 2 + w_c) / f_c))
)

z_deep = Constant(-720)
b = interpolate(max_value(B_x + B_y, z_deep), S)

# Some physical constants; `A = ε_c / τ_c ** n`, `C = τ_c / u_c ** (1 / m)`.
# Rather than work with the fluidity `A` and friction `C` directly, we use
# these stress, strain rate, and velocity scales so that we can easily rescale
# the physical constants under changing flow law and sliding exponents.
ε_c = Constant(0.02)
τ_c = Constant(0.1)
u_c = Constant(1000.0)

# Set up the boundary conditions.
inflow_ids = (1,)
outflow_ids = (2,)
side_wall_ids = (3, 4)

inflow_bc = firedrake.DirichletBC(Z.sub(0), Constant((0, 0)), inflow_ids)
side_wall_bc = firedrake.DirichletBC(Z.sub(0).sub(1), 0, side_wall_ids)
bcs = [inflow_bc, side_wall_bc]

# Set up the solution variables, input data, Lagrangian, and solvers for the
# momentum conservation equation.
fns = [
    ice_stream.viscous_power,
    ice_stream.friction_power,
    ice_stream.boundary,
    ice_stream.constraint,
]

z = firedrake.Function(Z)
δu = Constant(90)
z.sub(0).interpolate(as_vector((δu * x / Lx, 0)))

u, M, τ = firedrake.split(z)
h_0 = Constant(100.0)
h = interpolate(h_0, Q)
s = interpolate(max_value(b + h, (1 - ρ_I / ρ_W) * h), Q)
h_min = Constant(10.0)
p_I = ρ_I * g * max_value(h_min, h)
p_W = -ρ_W * g * min_value(0, s - h)
f = (1 - p_W / p_I) ** m
kwargs = {
    "velocity": u,
    "membrane_stress": M,
    "basal_stress": τ,
    "thickness": h,
    "surface": s,
    "floating": f,
    "viscous_yield_strain": ε_c,
    "viscous_yield_stress": τ_c,
    "friction_yield_speed": u_c,
    "friction_yield_stress": τ_c,
    "outflow_ids": outflow_ids,
}

linear_exponents = {"flow_law_exponent": 1, "sliding_law_exponent": 1}
L_1 = sum(fn(**kwargs, **linear_exponents) for fn in fns)
F_1 = firedrake.derivative(L_1, z)
firedrake.solve(F_1 == 0, z, bcs)

exponents = {"flow_law_exponent": n, "sliding_law_exponent": m}
L = sum(fn(**kwargs, **exponents) for fn in fns)
F = firedrake.derivative(L, z)

J_1 = firedrake.derivative(F_1, z)
J = firedrake.derivative(F, z)

α = Constant(1e-3)
params = {
    "solver_parameters": {
        "snes_type": "newtonls",
        "ksp_type": "gmres",
        "pc_type": "lu",
        "pc_factor_mat_solver_type": "mumps",
        "snes_max_it": 200,
    },
}
momentum_problem = NonlinearVariationalProblem(F, z, bcs, J=J + α * J_1)
momentum_solver = NonlinearVariationalSolver(momentum_problem, **params)
momentum_solver.solve()

# Set up the solution variables, input data, and solvers for the mass balance
# equation.
a = Constant(0.3)

dt = Constant(args.timestep)
φ = firedrake.TestFunction(Q)
ν = firedrake.FacetNormal(mesh)

G_cells = (Dt(h) * φ - inner(h * u, grad(φ)) - a * φ) * dx
f = h * max_value(0, inner(u, ν))
G_facets = (f("+") - f("-")) * (φ("+") - φ("-")) * dS
G_inflow = h_0 * min_value(0, inner(u, ν)) * φ * ds
G_outflow = f * φ * ds

G = G_cells + G_facets + G_inflow + G_outflow
tableau = irksome.BackwardEuler()
dt = Constant(args.timestep)
t = Constant(0.0)
mass_solver = irksome.TimeStepper(G, tableau, t, dt, h)

# Solve the coupled mass and momentum balance equations for several centuries.
with firedrake.CheckpointFile(args.output, "w") as chk:
    u, M, τ = z.subfunctions
    chk.save_function(u, name="velocity", idx=0)
    chk.save_function(M, name="membrane_stress", idx=0)
    chk.save_function(τ, name="basal_stress", idx=0)
    chk.save_function(h, name="thickness", idx=0)

    num_steps = int(args.final_time / float(dt))
    timesteps = np.linspace(0.0, args.final_time, num_steps + 1)
    progress_bar = tqdm.trange(num_steps)
    for step in progress_bar:
        try:
            mass_solver.advance()
            h.interpolate(max_value(0, h))

            s.interpolate(max_value(b + h, (1 - ρ_I / ρ_W) * h))
            momentum_solver.solve()
        except firedrake.ConvergenceError:
            chk.h5pyfile.create_dataset("timesteps", data=timesteps[:step])
            sys.exit()

        min_h = h.dat.data_ro.min()
        avg_h = firedrake.assemble(h * dx) / area
        description = f"avg, min h: {avg_h:4.2f}, {min_h:4.2f}"
        progress_bar.set_description(description)

        u, M, τ = z.subfunctions
        chk.save_function(u, name="velocity", idx=step + 1)
        chk.save_function(M, name="membrane_stress", idx=step + 1)
        chk.save_function(τ, name="basal_stress", idx=step + 1)
        chk.save_function(h, name="thickness", idx=step + 1)

    chk.h5pyfile.create_dataset("timesteps", data=timesteps)
