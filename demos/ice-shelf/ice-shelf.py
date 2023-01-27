import json
import argparse
import numpy as np
import firedrake
from firedrake import Constant, assemble, inner, dot, tr, grad, div, dx, ds
import icepack


parser = argparse.ArgumentParser()
parser.add_argument("--log-nx-min", type=int, default=4)
parser.add_argument("--log-nx-max", type=int, default=8)
parser.add_argument("--num-steps", type=int, default=9)
parser.add_argument("--output", default="results.json")
args = parser.parse_args()

# Physical constants
# TODO: Copy the values from icepack
ρ_I = Constant(icepack.constants.ice_density)
ρ_W = Constant(icepack.constants.water_density)
ρ = ρ_I * (1 - ρ_I / ρ_W)
g = Constant(icepack.constants.gravity)

# At about -14C, with an applied stress of 100 kPa, the glacier will experience
# a strain rate of 10 (m / yr) / km at -14C.
τ = Constant(0.1)  # MPa
ε = Constant(0.01) # (km / yr) / km


def exact_velocity(x, inflow_velocity, inflow_thickness, thickness_change, length):
    u_0 = Constant(inflow_velocity)
    h_0 = Constant(inflow_thickness)
    dh = Constant(thickness_change)
    lx = Constant(length)

    n = Constant(icepack.constants.glen_flow_law)
    A = ε / τ**n
    u_0 = Constant(100.0)
    h_0, dh = Constant(500.0), Constant(100.0)
    ζ = A * (ρ * g * h_0 / 4) ** n
    ψ = 1 - (1 - (dh / h_0) * (x[0] / Lx)) ** (n + 1)
    du = ζ * ψ * Lx * (h_0 / dh) / (n + 1)
    return firedrake.as_vector((u_0 + du, 0.0))


def action(z, h, u_in, inflow_ids, exponent):
    u, M = firedrake.split(z)
    mesh = z.ufl_domain()
    ν = firedrake.FacetNormal(mesh)
    d = mesh.geometric_dimension()

    n = Constant(exponent)
    A = ε / τ**n

    M_2 = (inner(M, M) - tr(M) ** 2 / (d + 1)) / 2
    if exponent == 1:
        M_n = M_2
    else:
        M_n = M_2 ** ((n + 1) / 2)

    power = 2 * h * A / (n + 1) * M_n * dx
    constraint = inner(u, div(h * M) - 0.5 * ρ * g * grad(h**2)) * dx
    boundary = h * inner(dot(M, ν), u_in) * ds(inflow_ids)
    return power + constraint - boundary


def solve_dual_problem(z, *args):
    params = {
        "solver_parameters": {
            "snes_type": "newtontr",
            "ksp_type": "gmres",
            "pc_type": "lu",
            "pc_factor_mat_solver_type": "mumps",
        }
    }

    L = action(z, *args, exponent=1)
    F = firedrake.derivative(L, z)
    firedrake.solve(F == 0, z, **params)

    L = action(z, *args, exponent=icepack.constants.glen_flow_law)
    F = firedrake.derivative(L, z)
    firedrake.solve(F == 0, z, **params)


results = []
k_min, k_max, num_steps = args.log_nx_min, args.log_nx_max, args.num_steps
for nx in np.logspace(k_min, k_max, num_steps, base=2, dtype=int):
    # Create the mesh and function spaces
    Lx, Ly = 20e3, 20e3
    mesh = firedrake.RectangleMesh(nx, nx, Lx, Ly, diagonal="crossed")
    d = mesh.geometric_dimension()

    cg1 = firedrake.FiniteElement("CG", "triangle", 1)
    Q = firedrake.FunctionSpace(mesh, cg1)
    V = firedrake.VectorFunctionSpace(mesh, cg1)
    b3 = firedrake.FiniteElement("B", "triangle", 3)
    Σ = firedrake.TensorFunctionSpace(mesh, cg1 + b3, symmetry=True)
    Z = V * Σ

    # Create the thickness field and the exact solution for the velocity
    x = firedrake.SpatialCoordinate(mesh)
    h_0, dh = Constant(500.0), Constant(100.0)
    U_exact = exact_velocity(
        x,
        inflow_velocity=100.0,
        inflow_thickness=float(h_0),
        thickness_change=float(dh),
        length=float(Lx),
    )
    lx = Constant(Lx)
    h = firedrake.interpolate(h_0 - dh * x[0] / lx, Q)

    # TODO: Fix this! Use Nitsche on the side walls & outflow to fix `M`
    # inflow_ids = (1,)
    # outflow_ids = (2,)
    # side_wall_ids = (3, 4)
    inflow_ids = (1, 2, 3, 4)
    u_in = U_exact

    z = firedrake.Function(Z)
    solve_dual_problem(z, h, u_in, inflow_ids)

    # Check the relative accuracy of the solution
    u, M = z.split()
    u_exact = firedrake.interpolate(U_exact, V)
    error = firedrake.norm(u - u_exact) / firedrake.norm(u_exact)
    δx = mesh.cell_sizes.dat.data_ro.min()
    results.append((δx, error))
    print(".", end="", flush=True)

with open(args.output, "w") as output_file:
    json.dump(results, output_file)
