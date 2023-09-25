import json
import argparse
import tqdm
import numpy as np
import firedrake
from firedrake import interpolate, as_vector, max_value, Constant
from icepack.constants import (
    ice_density as ρ_I,
    water_density as ρ_W,
    gravity as g,
    glen_flow_law as n,
    weertman_sliding_law as m,
)
from dualform import ice_stream


parser = argparse.ArgumentParser()
parser.add_argument("--log-nx-min", type=int, default=4)
parser.add_argument("--log-nx-max", type=int, default=8)
parser.add_argument("--degree", type=int, default=1)
parser.add_argument("--num-steps", type=int, default=9)
parser.add_argument("--output", default="results.json")
args = parser.parse_args()

Lx, Ly = Constant(20e3), Constant(20e3)
h0, dh = Constant(500.0), Constant(100.0)
u_inflow = Constant(100.0)

τ_c = Constant(0.1)
ε_c = Constant(0.01)
A = ε_c / τ_c**n

height_above_flotation = Constant(10.0)
d = Constant(-ρ_I / ρ_W * (h0 - dh) + height_above_flotation)
ρ = Constant(ρ_I - ρ_W * d**2 / (h0 - dh) ** 2)

# We'll arbitrarily pick this to be the velocity, then we'll find a
# friction coefficient and surface elevation that makes this velocity
# an exact solution of the shelfy stream equations.
def exact_u(x):
    Z = A * (ρ * g * h0 / 4) ** n
    q = 1 - (1 - (dh / h0) * (x / Lx)) ** (n + 1)
    du = Z * q * Lx * (h0 / dh) / (n + 1)
    return u_inflow + du


# With this choice of friction coefficient, we can take the surface
# elevation to be a linear function of the horizontal coordinate and the
# velocity will be an exact solution of the shelfy stream equations.
β = Constant(0.5)
α = Constant(β * ρ / ρ_I * dh / Lx)

def friction(x):
    return α * (ρ_I * g * (h0 - dh * x / Lx)) * exact_u(x) ** (-1 / m)


errors = []
k_min, k_max, num_steps = args.log_nx_min, args.log_nx_max, args.num_steps
for nx in np.logspace(k_min, k_max, num_steps, base=2, dtype=int):
    mesh = firedrake.RectangleMesh(nx, nx, float(Lx), float(Ly), diagonal="crossed")
    x, y = firedrake.SpatialCoordinate(mesh)

    cg = firedrake.FiniteElement("CG", "triangle", args.degree)
    dg = firedrake.FiniteElement("DG", "triangle", args.degree - 1)
    Q = firedrake.FunctionSpace(mesh, cg)
    V = firedrake.VectorFunctionSpace(mesh, cg)
    Σ = firedrake.TensorFunctionSpace(mesh, dg, symmetry=True)
    # TODO: investigate using DG for this?
    T = firedrake.VectorFunctionSpace(mesh, cg)
    Z = V * Σ * T
    z = firedrake.Function(Z)
    z.sub(0).assign(Constant((u_inflow, 0)))

    u_exact = interpolate(as_vector((exact_u(x), 0)), V)

    h = interpolate(h0 - dh * x / Lx, Q)
    ds = (1 + β) * ρ / ρ_I * dh
    s = interpolate(d + h0 - dh + ds * (1 - x / Lx), Q)

    # TODO: adjust the yield stress so that this has a more sensible value
    C = interpolate(friction(x), Q)
    u_c = interpolate((τ_c / C)**m, Q)

    inflow_ids = (1,)
    outflow_ids = (2,)
    side_wall_ids = (3, 4)

    u, M, τ = firedrake.split(z)
    kwargs = {
        "velocity": u,
        "membrane_stress": M,
        "basal_stress": τ,
        "thickness": h,
        "surface": s,
        "viscous_yield_strain": ε_c,
        "viscous_yield_stress": τ_c,
        "friction_yield_speed": u_c,
        "friction_yield_stress": τ_c,
        "outflow_ids": outflow_ids,
    }

    inflow_bc = firedrake.DirichletBC(Z.sub(0), Constant((u_inflow, 0)), inflow_ids)
    side_wall_bc = firedrake.DirichletBC(Z.sub(0).sub(1), 0, side_wall_ids)
    bcs = [inflow_bc, side_wall_bc]

    fns = [
        ice_stream.viscous_power,
        ice_stream.friction_power,
        ice_stream.boundary,
        ice_stream.constraint,
    ]

    J_l = sum(
        fn(**kwargs, flow_law_exponent=1, sliding_law_exponent=1) for fn in fns
    )
    F_l = firedrake.derivative(J_l, z)
    firedrake.solve(F_l == 0, z, bcs=bcs)

    qdegree = max(8, args.degree ** n)
    params = {
        "form_compiler_parameters": {"quadrature_degree": qdegree}
    }

    num_continuation_steps = 4
    for t in np.linspace(0.0, 1.0, num_continuation_steps):
        exponents = {
            "flow_law_exponent": Constant((1 - t) + t * n),
            "sliding_law_exponent": Constant((1 - t) + t * m),
        }
        J = sum(fn(**kwargs, **exponents) for fn in fns)
        F = firedrake.derivative(J, z)
        firedrake.solve(F == 0, z, bcs=bcs, **params)

    u, M, τ = z.subfunctions
    error = firedrake.norm(u - u_exact) / firedrake.norm(u_exact)
    δx = mesh.cell_sizes.dat.data_ro.min()
    errors.append((δx, error))

try:
    with open(args.output, "r") as input_file:
        results = json.load(input_file)
except FileNotFoundError:
    results = {}

results.update({f"degree-{args.degree}": errors})
with open(args.output, "w") as output_file:
    json.dump(results, output_file)
