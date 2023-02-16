import json
import argparse
import numpy as np
import firedrake
from firedrake import Constant
from icepack.constants import ice_density, water_density, gravity, glen_flow_law
from dualform import ice_shelf


parser = argparse.ArgumentParser()
parser.add_argument("--log-nx-min", type=int, default=4)
parser.add_argument("--log-nx-max", type=int, default=8)
parser.add_argument("--degree", type=int, default=1)
parser.add_argument("--num-steps", type=int, default=9)
parser.add_argument("--output", default="results.json")
args = parser.parse_args()

# Physical constants
# TODO: Copy the values from icepack
ρ_I = Constant(ice_density)
ρ_W = Constant(water_density)
ρ = ρ_I * (1 - ρ_I / ρ_W)
g = Constant(gravity)

# At about -14C, with an applied stress of 100 kPa, the glacier will experience
# a strain rate of 10 (m / yr) / km at -14C.
τ_c = Constant(0.1)  # MPa
ε_c = Constant(0.01) # (km / yr) / km


def exact_velocity(x, inflow_velocity, inflow_thickness, thickness_change, length):
    u_0 = Constant(inflow_velocity)
    h_0 = Constant(inflow_thickness)
    dh = Constant(thickness_change)
    lx = Constant(length)

    n = Constant(glen_flow_law)
    A = ε_c / τ_c**n
    u_0 = Constant(100.0)
    h_0, dh = Constant(500.0), Constant(100.0)
    ζ = A * (ρ * g * h_0 / 4) ** n
    ψ = 1 - (1 - (dh / h_0) * (x[0] / lx)) ** (n + 1)
    du = ζ * ψ * lx * (h_0 / dh) / (n + 1)
    return firedrake.as_vector((u_0 + du, 0.0))


errors = []
k_min, k_max, num_steps = args.log_nx_min, args.log_nx_max, args.num_steps
for nx in np.logspace(k_min, k_max, num_steps, base=2, dtype=int):
    # Create the mesh and function spaces
    Lx, Ly = 20e3, 20e3
    mesh = firedrake.RectangleMesh(nx, nx, Lx, Ly, diagonal="crossed")
    d = mesh.geometric_dimension()

    cg_k = firedrake.FiniteElement("CG", "triangle", args.degree)
    b_k = firedrake.FiniteElement("B", "triangle", args.degree + 2)
    Q = firedrake.FunctionSpace(mesh, cg_k)
    V = firedrake.VectorFunctionSpace(mesh, cg_k)
    Σ = firedrake.TensorFunctionSpace(mesh, cg_k + b_k, symmetry=True)
    Z = V * Σ

    # Create the thickness field and the exact solution for the velocity
    x = firedrake.SpatialCoordinate(mesh)
    h_0, dh = firedrake.Constant(500.0), firedrake.Constant(100.0)
    U_exact = exact_velocity(
        x,
        inflow_velocity=100.0,
        inflow_thickness=float(h_0),
        thickness_change=float(dh),
        length=Lx,
    )
    lx = firedrake.Constant(Lx)
    h = firedrake.interpolate(h_0 - dh * x[0] / lx, Q)

    # TODO: Fix this! Use Nitsche on the side walls
    # inflow_ids = (1,)
    # outflow_ids = (2,)
    # side_wall_ids = (3, 4)
    inflow_ids = (1, 3, 4)
    outflow_ids = (2,)
    u_in = U_exact

    z = firedrake.Function(Z)
    u, M = firedrake.split(z)
    kwargs = {
        "velocity": u,
        "membrane_stress": M,
        "thickness": h,
        "viscous_yield_strain": ε_c,
        "viscous_yield_stress": τ_c,
        "inflow_ids": inflow_ids,
        "outflow_ids": outflow_ids,
        "velocity_in": u_in,
    }
    fns = [ice_shelf.viscous_power, ice_shelf.boundary, ice_shelf.constraint]
    J_l = sum(fn(**kwargs, exponent=1) for fn in fns)
    F_l = firedrake.derivative(J_l, z)

    J = sum(fn(**kwargs, exponent=glen_flow_law) for fn in fns)
    F = firedrake.derivative(J, z)

    params = {
        "solver_parameters": {
            "snes_type": "newtontr",
            "ksp_type": "gmres",
            "pc_type": "lu",
            "pc_factor_mat_solver_type": "mumps",
        }
    }
    firedrake.solve(F_l == 0, z, **params)
    firedrake.solve(F == 0, z, **params)

    # Check the relative accuracy of the solution
    u, M = z.subfunctions
    u_exact = firedrake.interpolate(U_exact, V)
    error = firedrake.norm(u - u_exact) / firedrake.norm(u_exact)
    δx = mesh.cell_sizes.dat.data_ro.min()
    errors.append((δx, error))
    print(".", end="", flush=True)

try:
    with open(args.output, "r") as input_file:
        results = json.load(input_file)
except FileNotFoundError:
    results = {}

results.update({f"degree-{args.degree}": errors})
with open(args.output, "w") as output_file:
    json.dump(results, output_file)
