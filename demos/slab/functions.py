from firedrake import *
import numpy as np
from scipy import optimize
from matplotlib import pylab as plt
import os
import re
import ast

params_nl = {
         "snes_monitor": None,
        #  "snes_linesearch_type": "l2",
         "snes_max_it": 200,
         "snes_rtol": 1.0e-8,
         "snes_atol": 0,
         "snes_stol": 0,
         "snes_converged_reason": None,
        #  "snes_linesearch_monitor": None,
         }

params_lu = {
         "ksp_type": "preonly",
         "ksp_monitor": None,
         "pc_type" : "lu",
         "mat_type" : "aij",
         "pc_factor_mat_solver_type": "mumps",
         "mat_mumps_icntl_14": 200,
         }

def params_fs(num_var):
    fields0 = ""
    for i in range(num_var - 1):
        fields0 += "%d," % i
    fields0 = fields0[:-1]
    fields1 = "%d" % (num_var-1)
    params_fs = {
                "mat_type": "matfree",
                "ksp_type": "fgmres",
                "ksp_max_it": 10,
                "pc_type": "fieldsplit",
                "pc_fieldsplit_type": "schur",
                "pc_fieldsplit_schur_fact_type": "full",
                "pc_fieldsplit_0_fields": fields0,
                "pc_fieldsplit_1_fields": fields1,
                "fieldsplit_0_ksp_type": "preonly",
                "fieldsplit_0_pc_type": "python",
                "fieldsplit_0_pc_python_type": "firedrake.AssembledPC",
                "fieldsplit_0_assembled_pc_type": "lu",
                "fieldsplit_0_assembled_pc_factor_mat_solver_type": "mumps",
                "fieldsplit_0_assembled_mat_mumps_icntl_14": 200,
                "mat_mumps_icntl_14": 200,
                "fieldsplit_1_ksp_type": "gmres",
                "fieldsplit_1_ksp_max_it": 1,
                "fieldsplit_1_ksp_convergence_test": "skip",
                "fieldsplit_1_pc_type": "none",
    }
    return params_fs

def get_default_parser():
    import argparse
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--N", type = int, required = True)
    parser.add_argument("--FE", type=str, default="P1P1", choices = ["P1P1","P2P2","P1P1DP0","P2P2DP1"])
    parser.add_argument("--H", type = float, default = 5e2)
    parser.add_argument("--alpha", type = float, default = 1)
    parser.add_argument("--H0", type = float, default = 1500)
    parser.add_argument("--A", type = float, default = 1e-24)
    parser.add_argument("--C", type = float, default = 1e6)
    parser.add_argument("--n", type = int, default = 3)
    parser.add_argument("--rhoi", type = float, default = 917.)
    parser.add_argument("--rhow", type = float, default = 1000.)
    parser.add_argument("--g", type = float, default = 9.81)

    # Specific to marine ice sheets
    parser.add_argument("--a", type = float, default = 1)

    parser.add_argument("--reg", type = float, default = 1e-18)
    parser.add_argument("--meshref", type = float, default = 3)
    parser.add_argument("--save", dest='save', action='store_true')
    parser.set_defaults(save=False)
    return parser

def primal_momentum(u, v, h, xg, A, C, n, m, rhoi, rhow, g, db_fcn, reg, reg_slide):

    mesh = u.ufl_domain()
    sigma = SpatialCoordinate(mesh)
    sigma = sigma[0]
    r = 1 + 1./n
    dbed = xg * db_fcn(sigma * xg)
    delta = 1 - rhoi/rhow

    F = (
        - 2 * A**(-1./n) * h * ((u.dx(0))**2 + reg**2)**((r-2)/2.) * u.dx(0) * v.dx(0) * dx
        + xg**(r-1) * 0.5 * delta * rhoi * g * h**2 * v * ds(2)
        - xg**r * C * (u**2 + reg_slide**2)**((m-1)/2.) * u * v * dx
        - xg**(r-1) * rhoi * g * h * (h.dx(0) + dbed) * v * dx
    )

    return F

def primal(z, A, C, n, m, rhoi, rhow, g, b_fcn, db_fcn, reg, reg_slide, a, sl):

    Z = z.function_space()
    mesh = z.ufl_domain()
    (u,h,xg) = split(z)
    (v,q,s) = split(TestFunction(Z))
    sigma = SpatialCoordinate(mesh)
    sigma = sigma[0]
    bed = b_fcn(sigma * xg)
    hf = rhow / rhoi * (Constant(sl) - bed)

    F = primal_momentum(u, v, h, xg, A, C, n, m, rhoi, rhow, g, db_fcn, reg, reg_slide)

    F += (
        (u * h.dx(0) + u.dx(0) * h) * q * dx
        - xg * a * q * dx
        + (h - hf) * s * ds(2)
    )

    return F

def dual_momentum_one(u, v, tau, mu, h, xg, A, C, n, m, rhoi, rhow, g, db_fcn, reg_slide):

    mesh = u.ufl_domain()
    sigma = SpatialCoordinate(mesh)
    sigma = sigma[0]
    r = 1 + 1./n
    dbed = xg * db_fcn(sigma * xg)
    delta = 1 - rhoi/rhow

    F = (
        - h * tau * v.dx(0) * dx
        + 0.5 * delta * rhoi * g * h**2 * v * ds(2)
        - xg * C * (u**2 + reg_slide**2)**((m-1)/2.) * u * v * dx
        - rhoi * g * h * (h.dx(0) + dbed) * v * dx
        + 0.5**n * A * xg * abs(tau)**(n - 1) * tau * mu * dx
        - u.dx(0) * mu * dx
    )

    return F

def dual_one(z, A, C, n, m, rhoi, rhow, g, b_fcn, db_fcn, reg, reg_slide, a, sl):

    Z = z.function_space()
    mesh = z.ufl_domain()
    (u,h,tau,xg) = split(z)
    (v,q,mu,s) = split(TestFunction(Z))
    sigma = SpatialCoordinate(mesh)
    sigma = sigma[0]
    bed = b_fcn(sigma * xg)
    hf = rhow / rhoi * (Constant(sl) - bed)

    F = dual_momentum_one(u, v, tau, mu, h, xg, A, C, n, m, rhoi, rhow, g, db_fcn, reg_slide)

    F += (
        (u * h.dx(0) + u.dx(0) * h) * q * dx
        - xg * a * q * dx
        + (h - hf) * s * ds(2)
    )

    return F

def recover_coordinates_1D(V):
    mesh = V.mesh()
    Vvec = VectorFunctionSpace(mesh, V.ufl_element().family(), V.ufl_element().degree())
    xV = interpolate(SpatialCoordinate(mesh), Vvec).dat.data
    indV = np.argsort(xV)
    xV = xV[indV]
    return indV, xV

def path_name(args, type = "slab"):
    path = "output/" + type + "_" + args.FE + "_"
    path += "n%d_N%d_H%d_Ho%d_alpha%.4f_A%.e_C%.e" % (args.n, args.N, args.H, args.H0, args.alpha, args.A, args.C)
    if args.FE == "P2P2" or args.FE == "P1P1":
        path += "_reg%.e" % args.reg
    if ((type == "inclined") or (type == "schoof")):
        path += "_a%.2f" % (args.a * 3600 * 24 * 365)
    return path

def save_primal(z, path, args, bed, fnorm, xc, hc, uc):

    # make directory (check if already exists!)
    iter = 0
    path_new = path
    while os.path.exists(path_new):
        path_new = path + ("_alt%d" % iter)
        iter += 1
    if iter > 0:
        path = path_new
    os.mkdir(path)

    # Save run info
    f = open(path + "/run_info.txt","w")
    f.write(str(vars(args)))
    f.close()

    # Save solver info
    np.savetxt(path + "/newton_norm.txt", fnorm)

    # obtain (dimensional) arrays of DoFs ordered from left to right
    u, h = z.split()[:2]
    xg = z.split()[-1]
    xg = xg.dat.data[0] * xc
    indV, xV = recover_coordinates_1D(u.function_space())
    indH, xH = recover_coordinates_1D(h.function_space())
    udof = u.dat.data[indV] * uc
    hdof = h.dat.data[indH] * hc

    np.savetxt(path + "/xV.txt", xV * xg)
    np.savetxt(path + "/xH.txt", xH * xg)
    np.savetxt(path + "/u.txt", udof)
    np.savetxt(path + "/h.txt", hdof)
    np.savetxt(path + "/xg.txt", [xg])

    # plot dH
    xH *= xg
    dh = (hdof[1:] - hdof[:-1])/(xH[1:] - xH[:-1])
    xHm = 0.5 * (xH[1:] + xH[:-1])
    fig, ax = plt.subplots()
    ax.plot(xHm, dh)
    fig.savefig(path + "/dh.png",dpi = 300, bbox_inches = 'tight', pad_inches = 0.02)
    plt.close(fig)
    return path

    # # plot
    # figname = path + "/steady_state.png"
    # sl = 0
    # Lshelf = args.H0/np.tan(args.alpha * np.pi/180.)
    # q = udof[-1] * hdof[-1]
    # plot_surfaces(xV, udof, xH, xg, hdof, bed, Lshelf, sl, args.rhoi, args.rhow, args.g, q, args.A, args.n, figname)




def steady_ice_shelf(xg, bxg, rhoi, rhow, g, Ag, ng, q, sl, Lshelf, Nshelf):
    xshelf = np.linspace(xg, xg + Lshelf, Nshelf)
    Hf = rhow / rhoi * (sl - bxg)
    coef_beta = 2 * Ag**(-1./ng)
    coef_gamma = rhoi * (1-rhoi/rhow) * g
    coef_theta = 2 * coef_beta / coef_gamma
    coef1 = (ng + 1) * (q/coef_theta)**ng
    coef2 = (q/Hf)**(ng+1)
    u_SSA = (coef1 * (xshelf - xg) + coef2)**(1./(ng + 1))
    H_SSA = q / u_SSA
    theta_SSA = sl - rhoi/rhow * H_SSA
    ztop_SSA = theta_SSA + H_SSA
    return xshelf, theta_SSA, ztop_SSA


def plot_surfaces(xV, u_dof, x, xg, h_dof, bed_fcn, Lshelf, sl, rhoi, rhow, g, q, Ag, ng, figname):

    x = x * xg
    xV = xV * xg

    fig, ax = plt.subplots(3, figsize=(8,7))

    # plot bed
    xbed = np.linspace(x[0],x[-1] + Lshelf, 1000)
    ax[0].plot(xbed, bed_fcn(xbed), linewidth = 2, color = "black" )

    # plot grounded part
    ztop = h_dof + bed_fcn(x)
    ax[0].plot(x, ztop, linewidth = 2, color = "blue" )
    ax[0].plot(x, bed_fcn(x), linewidth = 2, color = "blue" )

    # plot floating part
    Nshelf = 1000
    bxg = bed_fcn(xg)
    xshelf, theta_SSA, ztop_SSA = steady_ice_shelf(xg, bxg, rhoi, rhow, g, Ag, ng, q, sl, Lshelf, Nshelf)

    ax[0].plot(xshelf, ztop_SSA, linewidth = 2, color = "blue" )
    ax[0].plot(xshelf, theta_SSA, linewidth = 2, color = "blue" )

    ax[0].plot([0,x[-1] + Lshelf], [sl,sl], linestyle = "--", color = "black")

    ax[0].set_xlim([xbed[0], xbed[-1]])
    ax[0].set_ylim([bed_fcn(xbed[-1]) ,1.1 * ztop[0]])

    # plot velocity
    ax[1].plot(xV, u_dof, linewidth = 1, color = "blue" )
    ax[1].set_xlim([xbed[0], xbed[-1]])
    ax[1].set_ylabel("$u$")

    ax[2].plot(x, h_dof, linewidth = 1, color = "blue" )
    ax[2].set_xlim([xbed[0], xbed[-1]])
    ax[2].set_ylabel("$H$")

    fig.savefig(figname,dpi = 300, bbox_inches = 'tight', pad_inches = 0.02)
    plt.close(fig)

def SIAapprox(A,C,n,a,rho,g,rhow,bed,dbed,beta,gamma,xga):

    # Find xg
    def param_func(x):
        return (a/beta)**(1./gamma) * rho/rhow * x**(1./gamma) + bed(x)
    xg = optimize.fsolve(param_func, xga)[0]
    res = param_func(xg)
    print("Approximation of grounding line via parametrisation :: %.2f " % xg)
    print("Residual of parametrised eq :: %.2e" % res)

    # solve Hsia
    N = 1000
    x = np.linspace(0,1,N) * xg
    xrev = x[::-1]
    dx = x[1:] - x[:-1]

    def func(h):
        return rho * g * h * (h/C**n + 2./(n + 2) * A * h**2)**(1./n)

    Hsia = np.zeros(N)
    Hsia[0] = -bed(xg) * rhow/rho
    for i in range(N-1):
        q0 = a * xrev[i]
        Hsia[i+1] = Hsia[i] + dx[i] * (dbed(xrev[i]) +  q0**(1./n)/func(Hsia[i]))
    Hsia = Hsia[::-1]
    usia = a * x / Hsia

    return x, Hsia, usia, xg
