import argparse
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib_scalebar.scalebar import ScaleBar
import firedrake
from firedrake import assemble, inner, sqrt, dx, ds
try:
    from firedrake.pyplot import tripcolor, streamplot
except ModuleNotFoundError:
    from firedrake import tripcolor, streamplot

parser = argparse.ArgumentParser()
parser.add_argument("--calving-freq", type=float)
args = parser.parse_args()

# Load the calving results
with firedrake.CheckpointFile("calving.h5", "r") as chk:
    timesteps_3 = chk.h5pyfile["timesteps"][:]
    mesh = chk.load_mesh()
    num_steps = len(timesteps_3)
    hs = [chk.load_function(mesh, "thickness", idx) for idx in range(num_steps)]
    us = [chk.load_function(mesh, "velocity", idx) for idx in range(num_steps)]
    Ms = [chk.load_function(mesh, "membrane_stress", idx) for idx in range(num_steps)]

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
fig, axes = plt.subplots(nrows=2, ncols=3, sharex=True, sharey=True, layout="compressed")
for ax in axes.flatten():
    ax.set_aspect("equal")
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

axes[0, 0].set_title("Thickness")
kw = {"xy": (0.02, 0.08), "xycoords": "axes fraction"}
axes[0, 0].annotate("a)", **kw)
colors = tripcolor(h_steady, vmin=0.0, axes=axes[0, 0])
fig.colorbar(colors, label="meters", orientation="horizontal", pad=0.04, ax=axes[0, 0])

axes[0, 1].set_title("Velocity")
axes[0, 1].annotate("b)", **kw)
colors = streamplot(u_steady, resolution=10e3, seed=1729, axes=axes[0, 1])
fig.colorbar(colors, label="meters/year", orientation="horizontal", pad=0.04, ax=axes[0, 1])

axes[0, 2].set_title("Membrane stress")
axes[0, 2].annotate("c)", **kw)
elt = firedrake.FiniteElement("DG", "triangle", M_steady.ufl_element().degree())
S = firedrake.FunctionSpace(mesh, elt)
m = firedrake.Function(S).interpolate(1e3 * sqrt(inner(Ms[0], Ms[0])))
colors = tripcolor(m, axes=axes[0, 2])
fig.colorbar(colors, label="kPa", orientation="horizontal", pad=0.04, ax=axes[0, 2])

axes[1, 0].set_title("Calved thickness")
axes[1, 0].annotate("d)", **kw)
index = (timesteps_3 <= args.calving_freq).argmin() + 1
colors = tripcolor(hs[index], vmin=0.0, axes=axes[1, 0])

axes[1, 1].set_title("Speed change")
axes[1, 1].annotate("e)", **kw)
V = us[0].function_space()
δu = firedrake.Function(V).interpolate(us[index] - us[0])
colors = firedrake.tripcolor(δu, axes=axes[1, 1])
fig.colorbar(colors, label="meters/year", orientation="horizontal", pad=0.04, ax=axes[1, 1])

axes[1, 2].set_title("Stress change")
axes[1, 2].annotate("f)", **kw)
ΔM = Ms[index] - Ms[0]
δM = firedrake.Function(S).project(1e3 * sqrt(inner(ΔM, ΔM)))
colors = firedrake.tripcolor(δM, axes=axes[1, 2])
fig.colorbar(colors, label="kPa", orientation="horizontal", pad=0.04, ax=axes[1, 2])

scalebar = ScaleBar(1, units="m", length_fraction=0.25, location="lower right")
axes[0, 0].add_artist(scalebar)
fig.get_layout_engine().set(h_pad=8/72)
fig.savefig("gibbous.pdf", bbox_inches="tight")


# Make a plot showing the volumes through the entire experiment
t1 = timesteps_1[-1]
t2 = timesteps_1[-1] + timesteps_2[-1]
t3 = t2 + timesteps_3[-1]
timesteps = np.concatenate((timesteps_1, timesteps_2 + t1, timesteps_3 + t2))
hs_all = hs_1 + hs_2 + hs
volumes = np.array([firedrake.assemble(h * dx) / 1e9 for h in hs_all])

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


# Make a plot showing the number of Newton iterations required during the
# calving phase of the experiment
with open("primal-counts.json", "r") as input_file:
    primal_counts = json.load(input_file)

with open("dual-counts.json", "r") as input_file:
    dual_counts = json.load(input_file)

fig, ax = plt.subplots(figsize=(6.4, 3.2))
indices = np.array(list(range(len(primal_counts))))
width = 0.25
ax.bar(indices - 2 * width, primal_counts, width=width, label="primal")
ax.bar(indices, dual_counts, width=width, label="dual")
ax.set_xlabel("Timestep (years)")
ax.set_ylabel("Iterations")
fig.savefig("counts.pdf", bbox_inches="tight")
