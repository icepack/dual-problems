import argparse
import numpy as np
import matplotlib.pyplot as plt
import firedrake
from firedrake import inner, sqrt


parser = argparse.ArgumentParser()
parser.add_argument("--input")
parser.add_argument("--output")
args = parser.parse_args()

with firedrake.CheckpointFile(args.input, "r") as chk:
    mesh = chk.load_mesh()
    timesteps = np.array(chk.h5pyfile["timesteps"])
    h = chk.load_function(mesh, "thickness", idx=len(timesteps) - 1)
    u = chk.load_function(mesh, "velocity", idx=len(timesteps) - 1)
    M = chk.load_function(mesh, "membrane_stress", idx=len(timesteps) - 1)
    τ = chk.load_function(mesh, "basal_stress", idx=len(timesteps) - 1)

fig, axes = plt.subplots(nrows=4, ncols=1, sharex=True, sharey=True)

for ax in axes:
    ax.set_aspect("equal")
    ax.set_xlim((0, 640e3))
    ax.set_ylim((0, 80e3))

for ax in axes[:-1]:
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

axes[-1].set_xlabel("easting (meters)")
axes[-1].set_ylabel("northing")
axes[-1].ticklabel_format(style="scientific", axis="both", scilimits=(0, 0))

axes[0].set_title("Thickness")
colors = firedrake.tripcolor(h, axes=axes[0])
fig.colorbar(colors, fraction=0.046, pad=0.04, shrink=0.8, ax=axes[0], label="meters")

axes[1].set_title("Velocity")
colors = firedrake.tripcolor(u, axes=axes[1])
fig.colorbar(colors, fraction=0.046, pad=0.04, shrink=0.8, ax=axes[1], label="meters/year")

axes[2].set_title("Membrane stress")
S = firedrake.FunctionSpace(mesh, "DG", M.ufl_element().degree())
m = firedrake.project(sqrt(inner(M, M)), S)
colors = firedrake.tripcolor(m, axes=axes[2])
fig.colorbar(colors, fraction=0.046, pad=0.04, shrink=0.8, ax=axes[2], label="megapascals")

axes[3].set_title("Basal stress")
colors = firedrake.tripcolor(τ, axes=axes[3])
fig.colorbar(colors, fraction=0.046, pad=0.04, shrink=0.8, ax=axes[3])

fig.savefig(args.output, bbox_inches="tight")
