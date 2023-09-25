import numpy as np
from firedrake import *
from functions import get_default_parser, recover_coordinates_1D

parser = get_default_parser()
args, _ = parser.parse_known_args()

### non-dimensionalise parameters
import argparse
args_dim = argparse.Namespace(**vars(args))
hc = args_dim.H
xc = args_dim.H
uc = (args_dim.rhoi * args_dim.g * args_dim.H / args_dim.C * np.sin(args_dim.alpha/180. * np.pi))**args_dim.n
sigmac = args_dim.rhoi * args_dim.g * args_dim.H
args.H = 1.0
args.H0 = args_dim.H0/xc
args.A = args_dim.A * args_dim.H * (args_dim.C/np.sin(args_dim.alpha/180. * np.pi))**args_dim.n
args.C = np.sin(args_dim.alpha/180. * np.pi)
args.rhoi = 1
args.g = 1
args.rhow = args_dim.rhow/args_dim.rhoi
args.reg = args_dim.reg * (xc / uc)**2

### Prepare geometry and mesh
alpha = args.alpha/180. * np.pi
H = args.H / np.cos(alpha)
mesh = UnitIntervalMesh(args.N)
m = 1./args.n

# refine towards the end
xU = mesh.coordinates.dat.data
# xU_new = (np.exp(-args.meshref* xU) - 1)/(np.exp(-args.meshref) - 1)
# mesh.coordinates.dat.data[:] = xU_new

# set function spaces
u0 = (args.rhoi * args.g * H / args.C * np.tan(alpha))**args.n
xg0 = (args.H0 + args.rhoi/args.rhow * H) / np.tan(alpha)

if args.FE == "P2P2" or args.FE == "P1P1":
    if args.FE == "P2P2":
        k = 2
    elif  args.FE == "P1P1":
        k = 1

    V = FunctionSpace(mesh, "CG", k)
    Q = FunctionSpace(mesh, "CG", k)
    R = FunctionSpace(mesh, "R", 0)
    Z = MixedFunctionSpace([V,Q,R])
    z = Function(Z)

    # set initial conditions
    z.sub(0).interpolate(Constant(u0))
    z.sub(1).interpolate(Constant(H))
    z.sub(2).assign(Constant(xg0))

    # set residual
    from functions import primal, save_primal
    residual = primal
    save = save_primal

elif args.FE == "P1P1DP0" or args.FE == "P2P2DP1":
    if args.FE == "P1P1DP0":
        k = 1
    elif  args.FE == "P2P2DP1":
        k = 2

    V = FunctionSpace(mesh, "CG", k)
    Q = FunctionSpace(mesh, "CG", k)
    S = FunctionSpace(mesh, "DG", k-1)
    R = FunctionSpace(mesh, "R", 0)
    Z = MixedFunctionSpace([V,Q,S,R])
    z = Function(Z)

    # set initial conditions
    z.sub(0).interpolate(Constant(u0))
    z.sub(1).interpolate(Constant(H))
    z.sub(3).assign(Constant(xg0))

    # set residual
    from functions import dual_one, save_primal
    residual = dual_one
    save = save_primal

else:
    raise ValueError("Finite element " + args.FE + " not available")

# set residual

# boundary conditions (only for the thickness)
bc = [DirichletBC(Z.sub(1), Constant(H), 1)]

# bed function
def b_fcn(x):
    return args.H0 - x * np.tan(alpha)
def b_fcn_dim(x):
    return args_dim.H0 - x * np.tan(alpha)
def db_fcn(x):
    return - np.tan(alpha)

# set solver parameters
from functions import params_nl, params_fs
params = {**params_nl, **params_fs(Z.num_sub_spaces())}

# solve problem
print("Solving steady problem...")
F = residual(z, args.A, args.C, args.n, m, args.rhoi, args.rhow, args.g,
            b_fcn, db_fcn, args.reg, 0, Constant(0), 0)
problem = NonlinearVariationalProblem(F, z, bcs = bc)
solver = NonlinearVariationalSolver(problem, solver_parameters = params)
solver.snes.setConvergenceHistory()
solver.solve()
u, h = z.split()[:2]
xg = z.split()[-1]

print("grounding line at x = %.4e" % (xg.dat.data[0] * xc))

# compare incoming velocity u0 to velocity at x = 0
indV, xV = recover_coordinates_1D(V)
u0_SSA = u.dat.data[indV[0]]
print("u0 = %.8e  m/a (numerical)" % (u0_SSA * uc * (3600 * 24 * 365)))
print("u0 = %.8e  m/a (analytical)" % (u0 * uc * (3600 * 24 * 365)))

# check flotation thickness
indQ, xQ = recover_coordinates_1D(Q)
print("flotation thickness :: %.4e (m)" % (h.dat.data[indQ[-1]] * xc))

# get solver info
fnorm, it = solver.snes.getConvergenceHistory()

# save
if args.save:
    print("Saving files...")
    from functions import path_name
    path = path_name(args_dim)
    save_primal(z, path, args_dim, b_fcn_dim, fnorm, xc, hc, uc)
