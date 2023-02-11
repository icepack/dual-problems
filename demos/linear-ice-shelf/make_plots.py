import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors

parser = argparse.ArgumentParser()
parser.add_argument("--input", default="results.json")
parser.add_argument("--output", default="results.eps")
args = parser.parse_args()

with open(args.input, "r") as input_file:
    data = json.load(input_file)

fig, ax = plt.subplots(constrained_layout=True)
ax.set_title("Convergence rate for ice shelf test case")
ax.set_xlabel("Mesh spacing (meters)")
ax.set_ylabel("$L^2$-norm relative error")
ax.set_xscale("log")
ax.set_yscale("log")

colors = list(matplotlib.colors.TABLEAU_COLORS.keys())
for name, results in data.items():
    results = np.array(results)
    mesh_sizes = results[:, 0]
    errors = results[:, 1]

    slope, intercept = np.polyfit(np.log(mesh_sizes), np.log(errors), 1)
    poly = np.poly1d((slope, intercept))
    log_dx_min = np.log(mesh_sizes).min()
    log_dx_max = np.log(mesh_sizes).max()
    dxs = np.logspace(log_dx_min - 0.25, log_dx_max + 0.25, 20, base=np.exp(1))
    errs = np.exp(poly(np.log(dxs)))

    color = colors.pop(0)
    ax.scatter(mesh_sizes, errors, color=color)
    label = f"{name}, error ~ $\delta x^{{{slope:2.1f}}}$"
    ax.plot(dxs, errs, color=color, label=label)

ax.legend()
fig.savefig(args.output, bbox_inches="tight")
