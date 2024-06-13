import json
import argparse
import numpy as np
import firedrake
from firedrake import Constant
from icepack2.constants import (
    ice_density as ρ_I,
    water_density as ρ_W,
    gravity as g,
    glen_flow_law as n,
)
from icepack2 import model


parser = argparse.ArgumentParser()
parser.add_argument("--log-nx-min", type=int, default=4)
parser.add_argument("--log-nx-max", type=int, default=8)
parser.add_argument("--degree", type=int, default=1)
parser.add_argument("--num-steps", type=int, default=9)
parser.add_argument("--output", default="results.json")
args = parser.parse_args()

# Physical constants at about -14C, with an applied stress of 100 kPa, the
# glacier will experience a strain rate of 10 (m / yr) / km at -14C.
ρ = ρ_I * (1 - ρ_I / ρ_W)
τ_c = Constant(0.1)  # MPa
ε_c = Constant(0.01) # (km / yr) / km


def exact_velocity(x, inflow_velocity, inflow_thickness, thickness_change, length):
    u_0 = Constant(inflow_velocity)
    h_0 = Constant(inflow_thickness)
    dh = Constant(thickness_change)
    lx = Constant(length)

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

    cg = firedrake.FiniteElement("CG", "triangle", args.degree)
    dg = firedrake.FiniteElement("DG", "triangle", args.degree - 1)
    Q = firedrake.FunctionSpace(mesh, cg)
    V = firedrake.VectorFunctionSpace(mesh, cg)
    Σ = firedrake.TensorFunctionSpace(mesh, dg, symmetry=True)
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
    h = firedrake.Function(Q).interpolate(h_0 - dh * x[0] / lx)

    inflow_ids = (1,)
    outflow_ids = (2,)
    side_wall_ids = (3, 4)
    u_in = U_exact

    z = firedrake.Function(Z)
    u, M = firedrake.split(z)
    fields = {
        "velocity": u,
        "membrane_stress": M,
        "thickness": h,
    }
    rheology = {
        "flow_law_exponent": n,
        "flow_law_coefficient": ε_c / τ_c**n,
    }
    linear_rheology = {
        "flow_law_exponent": 1,
        "flow_law_coefficient": ε_c / τ_c,
    }
    fns = [model.viscous_power, model.ice_shelf_momentum_balance]
    J_l = sum(fn(**fields, **linear_rheology) for fn in fns)
    F_l = firedrake.derivative(J_l, z)

    J = sum(fn(**fields, **rheology) for fn in fns)
    F = firedrake.derivative(J, z)

    inflow_bc = firedrake.DirichletBC(Z.sub(0), u_in, inflow_ids)
    side_wall_bc = firedrake.DirichletBC(Z.sub(0).sub(1), 0, side_wall_ids)

    params = {
        "solver_parameters": {
            "snes_type": "newtontr",
            "ksp_type": "gmres",
            "pc_type": "lu",
            "pc_factor_mat_solver_type": "mumps",
        },
        "bcs": [inflow_bc, side_wall_bc],
    }
    firedrake.solve(F_l == 0, z, **params)
    firedrake.solve(F == 0, z, **params)

    # Check the relative accuracy of the solution
    u, M = z.subfunctions
    u_exact = firedrake.Function(V).interpolate(U_exact)
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
