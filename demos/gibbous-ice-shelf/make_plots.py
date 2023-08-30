import argparse
import firedrake
from firedrake import inner, sqrt
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument("--input", default="gibbous.h5")
parser.add_argument("--output", default="gibbous.png")
args = parser.parse_args()

with firedrake.CheckpointFile(args.input, "r") as chk:
    mesh = chk.load_mesh()
    u = chk.load_function(mesh, name="velocity")
    M = chk.load_function(mesh, name="membrane_stress")
    h = chk.load_function(mesh, name="thickness")

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
colors = firedrake.tripcolor(h, axes=axes[0])
fig.colorbar(colors, label="meters", orientation="horizontal", pad=0.04, ax=axes[0])

axes[1].set_title("Velocity")
colors = firedrake.streamplot(u, resolution=10e3, axes=axes[1])
fig.colorbar(colors, label="meters/year", orientation="horizontal", pad=0.04, ax=axes[1])

axes[2].set_title("Membrane stress")
elt = firedrake.FiniteElement("DG", "triangle", M.ufl_element().degree())
S = firedrake.FunctionSpace(mesh, elt)
m = firedrake.interpolate(1e3 * sqrt(inner(M, M)), S)
colors = firedrake.tripcolor(m, axes=axes[2])
fig.colorbar(colors, label="kPa", orientation="horizontal", pad=0.04, ax=axes[2])

fig.savefig(args.output, bbox_inches="tight")
