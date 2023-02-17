import os
import argparse
import ast
import numpy as np
import sys
sys.path.append("../VisCoSo/subglacialcavity/postprocess/jfm_paper")
from figure_settings import pgf_with_latex
from functions import path_name, get_default_parser, recover_coordinates_1D

k = [1]
reg = [1e-4, 1e-6, 1e-8, 1e-10, 1e-12, 1e-14, 1e-16]

# set parser and command
parser = get_default_parser()
args, _ = parser.parse_known_args()
command = "python slab.py --N %d --save --H0 %d --H %d --A %.e --C %.e --n %d " % (args.N, args.H0, args.H, args.A, args.C, args.n)

# prepare loop
args_copy = argparse.Namespace(**vars(args))
its = np.zeros([len(k), len(reg)])
min_norm = np.zeros([len(k), len(reg)])
max_norm = np.zeros([len(k), len(reg)])
xg = np.zeros([len(k), len(reg) + 1])
hxg = np.zeros([len(k), len(reg) + 1])

# set plot for fnorm
from matplotlib import pylab as plt
import matplotlib
matplotlib.rcParams.update(pgf_with_latex(1, hscale = 0.5))

fig, ax = plt.subplots(1,len(k))
cmap = matplotlib.cm.get_cmap('Reds')
colors = [cmap(i) for i in np.linspace(0.25,1,len(reg))]


for i, ki in enumerate(k):

    # set plot
    if len(k) == 1:
        axi = ax
    else:
        axi = ax[i]

    fnorm = []

    # solve for primal solver
    if ki == 2:
        args_copy.FE = "P2P2"
    elif ki > 2:
        raise ValueError("No solver for k > 2")
    for j, regj in enumerate(reg):
        args_copy.reg = regj/(3600 * 24 * 24)
        path = path_name(args_copy)
        print(path)
        if os.path.isdir(path):
            print("Reading stored data for primal solve for reg = %.0e" % regj)
        else:
            print("Running primal solver for reg = %.0e" % regj)
            command_run = command + "--reg %.0e " % args_copy.reg + "--FE " + args_copy.FE + " --alpha %.2f " % args.alpha
            os.system(command_run)

        # save data
        xg[i,j] = float(np.loadtxt(path + "/xg.txt"))/1000.
        hxg[i,j] = np.loadtxt(path + "/h.txt")[-1]
        fnorm.append(np.loadtxt(path + "/newton_norm.txt"))
        its[i,j] = len(fnorm[j])
        min_norm[i,j] = 10**(np.floor(np.log10(min(fnorm[j]))))
        max_norm[i,j] = 10**(np.ceil(np.log10(max(fnorm[j]))))

    # dual solver
    if ki == 1:
        args_copy.FE = "P1P1DP0"
    elif ki == 2:
        args_copy.FE = "P2P2DP1"
    path = path_name(args_copy)
    if os.path.isdir(path):
        print("Reading stored data for dual solve")
    else:
        print("Running dual solver")
        command_run = command + "--FE " + args_copy.FE + " --alpha %.2f " % args.alpha
        os.system(command_run)

    xg[i,-1] = float(np.loadtxt(path + "/xg.txt"))/1000.
    hxg[i,-1] = np.loadtxt(path + "/h.txt")[-1]
    fnorm.append(np.loadtxt(path + "/newton_norm.txt"))
    min_norm[i,-1] = 10**(np.floor(np.log10(min(fnorm[-1]))))
    max_norm[i,-1] = 10**(np.ceil(np.log10(max(fnorm[-1]))))
    its[i,-1] = len(fnorm[-1])

    # Find values of reg for which we obtain the same value of xg
    ind_acc = np.where(np.abs(xg[i,-1]-xg[i,:-1])/xg[i,-1] < 0.001)[0]

    # plot fnorm
    for i in range(len(fnorm)):
        if i == (len(fnorm) - 1):
            color = "black"
            marker = "o"
            label = "dual"
            lw = 2
            ms = 4
        else:
            color = colors[i]
            exp = np.log10(reg[i])
            label = label = "$\\epsilon = 10^{%d}$" % exp
            lw = 1
            ms = 2
            if i in ind_acc:
                marker = "s"
            else:
                marker = "o"
        axi.plot(fnorm[i], color = color, linewidth = lw, marker = marker, markersize = ms, label = label)
    axi.set_yscale('log')

maxits = np.max(its)
order = 10**(np.floor(np.log10(maxits)))
maxits = np.ceil(maxits/order) * order
min_norm = np.min(min_norm)
max_norm = np.max(max_norm)
for i in range(len(k)):
    if len(k) == 1:
        axi = ax
    else:
        axi = ax[i]
    axi.set_xlim([0,maxits])
    axi.set_ylim([min_norm,max_norm])
    axi.grid()
    axi.set_xlabel("iterations")
    axi.set_yscale('log')
    axi.set_xticks(np.linspace(0,maxits,5))
    if i == 0:
        axi.set_ylabel("Newton residual norm")
        axi.legend(fontsize = 6, ncol = 4, loc='upper center', bbox_to_anchor=(0.5, 1.25))

fig.savefig("figures/newton_its_alpha%.2f.pdf" % args.alpha, bbox_inches = 'tight', pad_inches = 0.02)
plt.close(fig)

# save table and plot
table_file = open("tables/table_alpha%.2f.txt" % args.alpha, "w")
line1 = "solver & $\\epsilon\\,\\si{a}^{-1}$"
for i in range(len(k)):
    line1 += " & $x_g\\,\\si{km}$ & $h(x_g)\\,\\si{m}$ & iterations "
line1 += "\\\\ \n"
table_file.write(line1)
for i in range(len(reg)):
    if i == 0:
        line = "\\multirow{%d}{*}{primal} & " % len(reg)
    else:
        line = " & "
    exp = np.log10(reg[i])
    line += "$10^{%d}$ " % exp
    for j in range(len(k)):
        line += "& %.2f & %.2f & %d " % (xg[j,i], hxg[j,i], its[j,i])
    line += " \\\\ \n"
    table_file.write(line)
line = "dual & - "
for j in range(len(k)):
    line += "& %.2f & %.2f & %d " % (xg[j,i], hxg[j,i], its[j,i])
table_file.write(line)
table_file.close()

# plot strain rates and geometry
ki = k[-1]
from firedrake import *
f = open(path + "/run_info.txt", "r")
contents = f.read()
run_info = ast.literal_eval(contents)
f.close()
N = run_info["N"]
xg = float(np.loadtxt(path + "/xg.txt"))
udof = np.loadtxt(path + "/u.txt")
xV = np.loadtxt(path + "/xV.txt")
mesh = IntervalMesh(run_info["N"], xg)
if run_info["FE"] == "P2P2DP1":
    indUV = 2 * np.array(range(N+1))
    V = FunctionSpace(mesh, "CG", 2)
elif run_info["FE"] == "P1P1DP0":
    indUV = np.array(range(N+1))
    V = FunctionSpace(mesh, "CG", 1)
mesh.coordinates.dat.data[:] = xV[indUV]
u = Function(V)
u.dat.data[:] = udof
M = FunctionSpace(mesh, "DG", 0)
indM, xM = recover_coordinates_1D(M)
strain = project(abs(u.dx(0)), M)
strain_dof = strain.dat.data * 3600 * 24 * 365
fig, ax = plt.subplots()
ax.plot(xM/xg, strain_dof, color = "black", linewidth = 2)
for i, regj in enumerate(reg):
    if i in ind_acc:
        linestyle = ":"
    else:
        linestyle = "-"
    ax.plot([0,1], [regj,regj], color = colors[i], linestyle = linestyle)

ax.set_xlim([0,1])
ax.set_yscale('log')
fig.savefig("figures/strain_k%d_alpha%.2f.pdf" % (ki, args.alpha), bbox_inches = 'tight', pad_inches = 0.02)
plt.close(fig)


# plot geometry
hdof = np.loadtxt(path + "/h.txt")
xH = np.loadtxt(path + "/xH.txt")
q = hdof[-1] * udof[-1]
Lshelf = xg
def bed(x):
    return args.H0 - x * np.tan(args.alpha * np.pi/180.)
from functions import steady_ice_shelf
xshelf, theta_SSA, ztop_SSA = steady_ice_shelf(xg, bed(xg), args.rhoi, args.rhow, args.g, args.A, args.n, q, 0, Lshelf, 100)
xall = np.append(xH,xshelf[1:])
theta = np.append(bed(xH), theta_SSA[1:])
ztop = np.append(bed(xH) + hdof, ztop_SSA[1:])
bed_points = bed(xall)
fig, ax = plt.subplots()
xall *= 1e-3
ax.plot(xall,np.zeros(xall.size), color = "black", linestyle = "--", linewidth = 1)
ax.plot(xall,bed_points, color = "black", linewidth = 1)
ax.plot(xall,theta, color = "black", linewidth = 2)
ax.plot(xall,ztop, color = "black", linewidth = 2)
ax.set_ylim([1.4 * min(theta), 1.1 * max(ztop)])
ax.set_xlim([min(xall), max(xall)])
ax.set_xlabel("(km)")
ax.set_ylabel("(m)")
fig.savefig("figures/marine_ice_sheet.pdf", bbox_inches = 'tight', pad_inches = 0.02)
