import argparse
import numpy as np
import matplotlib.pyplot as plt
import firedrake
from firedrake import assemble, inner, sqrt, dx, ds

parser = argparse.ArgumentParser()
parser.add_argument("--calving-freq", type=float)
args = parser.parse_args()

# Load the calving results
with firedrake.CheckpointFile("calving.h5", "r") as chk:
    timesteps_3 = chk.h5pyfile["timesteps"][:]
    mesh = chk.load_mesh()
    hs_3 = [
        chk.load_function(mesh, "thickness", idx)
        for idx in range(len(timesteps_3))
    ]

# Get the results from the initial coarse spin-up and project them into the
# function space we used for the calving experiment
with firedrake.CheckpointFile("steady-state-coarse.h5", "r") as chk:
    timesteps_1 = chk.h5pyfile["timesteps"][:]
    mesh_1 = chk.load_mesh()
    hs_1 = [
        chk.load_function(mesh_1, "thickness", idx)
        for idx in range(len(timesteps_1))
    ]

# Get the results from the secondary fine spin-up and project
with firedrake.CheckpointFile("steady-state-fine.h5", "r") as chk:
    timesteps_2 = chk.h5pyfile["timesteps"][:]
    mesh_2 = chk.load_mesh()
    hs_2 = [
        chk.load_function(mesh_2, "thickness", idx)
        for idx in range(len(timesteps_2))
    ]

    h_steady = chk.load_function(mesh_2, "thickness", len(timesteps_2) - 1)
    M_steady = chk.load_function(mesh_2, "membrane_stress", len(timesteps_2) - 1)
    u_steady = chk.load_function(mesh_2, "velocity", len(timesteps_2) - 1)

# Make a plot showing the steady state at the end of the final spin-up
fig, axes = plt.subplots(nrows=1, ncols=3, sharex=True, sharey=True)
for ax in axes:
    ax.set_aspect("equal")
for ax in axes[1:]:
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

axes[0].get_xaxis().set_visible(False)
axes[0].set_ylabel("northing (meters)")
axes[0].ticklabel_format(style="sci", axis="both", scilimits=(0, 0))

axes[0].set_title("Thickness")
colors = firedrake.tripcolor(h_steady, axes=axes[0])
fig.colorbar(colors, label="meters", orientation="horizontal", pad=0.04, ax=axes[0])

axes[1].set_title("Velocity")
colors = firedrake.streamplot(u_steady, resolution=10e3, axes=axes[1])
fig.colorbar(colors, label="meters/year", orientation="horizontal", pad=0.04, ax=axes[1])

axes[2].set_title("Membrane stress")
elt = firedrake.FiniteElement("DG", "triangle", M_steady.ufl_element().degree())
S = firedrake.FunctionSpace(M_steady.ufl_domain(), elt)
m = firedrake.interpolate(1e3 * sqrt(inner(M_steady, M_steady)), S)
colors = firedrake.tripcolor(m, axes=axes[2])
fig.colorbar(colors, label="kPa", orientation="horizontal", pad=0.04, ax=axes[2])

fig.savefig("steady-state.pdf", bbox_inches="tight")

# Make a plot showing the thickness immediately after calving
fig, axes = plt.subplots()
axes.set_aspect("equal")
axes.get_xaxis().set_visible(False)
axes.set_ylabel("northing (meters)")
axes.ticklabel_format(style="sci", axis="both", scilimits=(0, 0))
index = (timesteps_3 <= args.calving_freq).argmin() + 1
colors = firedrake.tripcolor(hs_3[index], axes=axes)
fig.colorbar(colors, label="meters", orientation="horizontal", pad=0.04)
fig.savefig("calved-thickness.pdf", bbox_inches="tight")

# Make a plot showing the volumes through the entire experiment
t1 = timesteps_1[-1]
t2 = timesteps_1[-1] + timesteps_2[-1]
t3 = t2 + timesteps_3[-1]
timesteps = np.concatenate((timesteps_1, timesteps_2 + t1, timesteps_3 + t2))
hs = hs_1 + hs_2 + hs_3
volumes = np.array([firedrake.assemble(h * dx) / 1e9 for h in hs])

fig, ax = plt.subplots(figsize=(6, 3.2))
ax.set_xlabel("Time (years)")
ax.set_ylabel("Ice volume (km${}^3$)")
ax.plot(timesteps, volumes)
ax.set_ylim((10.5e3, 12.15e3))
arrowprops = {"arrowstyle": "|-|", "color": "tab:grey"}
blevel = 11e3
bdelta = 0.08e3
tlevel = blevel - bdelta
tdelta = 5
ax.annotate("", xy=(tdelta, blevel), xytext=(t1 - tdelta, blevel), arrowprops=arrowprops)
ax.annotate("", xy=(t1 + tdelta, blevel), xytext=(t2 - tdelta, blevel), arrowprops=arrowprops)
ax.annotate("", xy=(t2 + tdelta, blevel), xytext=(t3 - tdelta, blevel), arrowprops=arrowprops)
kwargs = {"color": "tab:grey", "ha": "center"}
ax.annotate("coarse\nspin-up", xy=(0, tlevel), xytext=(t1 / 2, tlevel), **kwargs)
ax.annotate("fine\nspin-up", xy=(t1, tlevel), xytext=((t1 + t2) / 2, tlevel), **kwargs)
ax.annotate("calving\ncycle", xy=(t2, tlevel), xytext=((t2 + t3) / 2, tlevel), **kwargs)
fig.savefig("volumes.pdf", bbox_inches="tight")
