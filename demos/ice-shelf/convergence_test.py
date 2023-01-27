import json
import argparse
import numpy as np
import firedrake
import ice_shelf


parser = argparse.ArgumentParser()
parser.add_argument("--log-nx-min", type=int, default=4)
parser.add_argument("--log-nx-max", type=int, default=8)
parser.add_argument("--num-steps", type=int, default=9)
parser.add_argument("--output", default="results.json")
args = parser.parse_args()

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
    h_0, dh = firedrake.Constant(500.0), firedrake.Constant(100.0)
    U_exact = ice_shelf.exact_velocity(
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
    ice_shelf.solve_dual_problem(z, h, u_in, inflow_ids, outflow_ids)

    # Check the relative accuracy of the solution
    u, M = z.split()
    u_exact = firedrake.interpolate(U_exact, V)
    error = firedrake.norm(u - u_exact) / firedrake.norm(u_exact)
    δx = mesh.cell_sizes.dat.data_ro.min()
    results.append((δx, error))
    print(".", end="", flush=True)

with open(args.output, "w") as output_file:
    json.dump(results, output_file)
