import json
import numpy as np
from numpy import pi as π
import tqdm
import firedrake
from firedrake import inner, derivative
import irksome
from icepack2 import model, solvers
from icepack2.constants import glen_flow_law as n

# Load the input data
with firedrake.CheckpointFile("steady-state-coarse.h5", "r") as chk:
    mesh = chk.load_mesh()
    idx = len(chk.h5pyfile["timesteps"]) - 1
    h = chk.load_function(mesh, "thickness", idx=idx)
    h0 = h.copy(deepcopy=True)
    u = chk.load_function(mesh, "velocity", idx=idx)
    M = chk.load_function(mesh, "membrane_stress", idx=idx)

Q = h.function_space()
V = u.function_space()
Σ = M.function_space()

Z = V * Σ
z = firedrake.Function(Z)
z.sub(0).assign(u)
z.sub(1).assign(M);

# Set up the Lagrangian
ε_c = firedrake.Constant(0.01)
τ_c = firedrake.Constant(0.1)

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

# Set up a regularized Lagrangian
h_min = firedrake.Constant(0.1)
rfields = {
    "velocity": u,
    "membrane_stress": M,
    "thickness": firedrake.max_value(h_min, h),
}

L_r = sum(fn(**rfields, **rheology) for fn in fns)
F_r = derivative(L_r, z)
H_r = derivative(F_r, z)

# Set up the mass balance equation
prognostic_problem = model.mass_balance(
    thickness=h,
    velocity=u,
    accumulation=firedrake.Constant(0.0),
    thickness_inflow=h0,
    test_function=firedrake.TestFunction(Q),
)

final_time = 400.0
num_steps = 400

dt = firedrake.Constant(final_time / num_steps)
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

# Set up the diagnostic solver and boundary conditions and do an initial solve
bc = firedrake.DirichletBC(Z.sub(0), firedrake.Constant((0, 0)), (1,))
problem = solvers.ConstrainedOptimizationProblem(L, z, H=H_r, bcs=bc)
diagnostic_solver = solvers.NewtonSolver(problem, tolerance=1e-4)

residuals = [list(diagnostic_solver.solve())]

# Create the calving mask -- this describes how we'll remove ice
radius = firedrake.Constant(60e3)
x = firedrake.SpatialCoordinate(mesh)
y = firedrake.Constant((0.0, radius))
mask = firedrake.conditional(inner(x - y, x - y) < radius**2, 0.0, 1.0)

# Run the simulation
calving_frequency = 24.0
time_since_calving = 0.0

for step in tqdm.trange(num_steps):
    prognostic_solver.advance()

    if time_since_calving > calving_frequency:
        h.interpolate(mask * h)
        time_since_calving = 0.0
    time_since_calving += float(dt)
    h.interpolate(firedrake.max_value(0, h))

    residuals.append(list(diagnostic_solver.solve()))

# Save the results to disk
with open("residuals.json", "w") as residuals_file:
    json.dump(residuals, residuals_file)
