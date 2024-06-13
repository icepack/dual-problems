import argparse
import numpy as np
import tqdm
import firedrake
from firedrake import (
    derivative,
    NonlinearVariationalProblem,
    NonlinearVariationalSolver,
)
import icepack
import irksome
from icepack2 import model
from icepack2.constants import glen_flow_law as n
import gibbous_inputs

parser = argparse.ArgumentParser()
parser.add_argument("--resolution", type=float, default=5e3)
parser.add_argument("--final-time", type=float, default=400.0)
parser.add_argument("--num-steps", type=int, default=200)
parser.add_argument("--form", choices=["primal", "dual"])
args = parser.parse_args()

# Generate and load the mesh
R = 200e3
δx = args.resolution

mesh = gibbous_inputs.make_mesh(R, δx)

print(mesh.num_vertices(), mesh.num_cells())

h_expr, u_expr = gibbous_inputs.make_initial_data(mesh, R)

# Create some function spaces and some fields
cg = firedrake.FiniteElement("CG", "triangle", 1)
dg0 = firedrake.FiniteElement("DG", "triangle", 0)
dg1 = firedrake.FiniteElement("DG", "triangle", 1)
Q = firedrake.FunctionSpace(mesh, dg1)
V = firedrake.VectorFunctionSpace(mesh, cg)

h0 = firedrake.Function(Q).interpolate(h_expr)
u0 = firedrake.Function(V).interpolate(u_expr)

h = h0.copy(deepcopy=True)

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


if args.form == "primal":
    u = u0.copy(deepcopy=True)
    solver = icepack.solvers.FlowSolver(icepack.models.IceShelf(), dirichlet_ids=[1])
    A = firedrake.Constant(ε_c / τ_c ** n)
    solver.diagnostic_solve(velocity=u, thickness=h, fluidity=A)
    diagnostic_solver = solver._diagnostic_solver
elif args.form == "dual":
    Σ = firedrake.TensorFunctionSpace(mesh, dg0, symmetry=True)
    Z = V * Σ
    z = firedrake.Function(Z)

    # Set up the diagnostic problem
    u, M = firedrake.split(z)
    fields = {
        "velocity": u,
        "membrane_stress": M,
        "thickness": h,
    }

    fns = [model.viscous_power, model.ice_shelf_momentum_balance]

    rheology = {
        "flow_law_exponent": n,
        "flow_law_coefficient": ε_c / τ_c ** n,
    }

    L = sum(fn(**fields, **rheology) for fn in fns)
    F = derivative(L, z)

    # Make an initial guess by solving a Picard linearization
    linear_rheology = {
        "flow_law_exponent": 1,
        "flow_law_coefficient": ε_c / τ_c,
    }

    L_1 = sum(fn(**fields, **linear_rheology) for fn in fns)
    F_1 = derivative(L_1, z)

    qdegree = n + 2
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

    # Create an approximate linearization
    h_min = firedrake.Constant(1.0)
    rfields = {
        "velocity": u,
        "membrane_stress": M,
        "thickness": firedrake.max_value(h_min, h),
    }

    L_r = sum(fn(**rfields, **rheology) for fn in fns)
    F_r = derivative(L_r, z)
    J_r = derivative(F_r, z)

    # Set up the diagnostic problem and solver
    diagnostic_problem = NonlinearVariationalProblem(F, z, J=J_r, **problem_params)
    diagnostic_solver = NonlinearVariationalSolver(diagnostic_problem, **solver_params)
    diagnostic_solver.solve()
else:
    raise ValueError(f"--form must be either `primal` or `dual`, got {args.form}")

# Set up the prognostic problem and solver
prognostic_problem = model.mass_balance(
    thickness=h,
    velocity=u,
    accumulation=firedrake.Constant(0.0),
    thickness_inflow=h0,
    test_function=firedrake.TestFunction(Q),
)
dt = firedrake.Constant(args.final_time / args.num_steps)
t = firedrake.Constant(0.0)
method = irksome.BackwardEuler()
prognostic_params = {
    "solver_parameters": {
        "snes_type": "ksponly",
        "ksp_type": "gmres",
        "pc_type": "bjacobi",
    },
}
prognostic_solver = irksome.TimeStepper(
    prognostic_problem, method, t, dt, h, **prognostic_params
)

# Run the simulation
for step in tqdm.trange(args.num_steps):
    prognostic_solver.advance()
    h.interpolate(firedrake.max_value(0, h))
    diagnostic_solver.solve()
