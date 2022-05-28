import json
import argparse
import numpy as np
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument("--input", default="results.json")
parser.add_argument("--output", default="results.eps")
args = parser.parse_args()

with open(args.input, "r") as input_file:
    results = np.array(json.load(input_file))

mesh_sizes = results[:, 0]
errors = results[:, 1]

slope, intercept = np.polyfit(np.log(mesh_sizes), np.log(errors), 1)
poly = np.poly1d((slope, intercept))
dxs = np.logspace(np.log(mesh_sizes).min() - 0.25, np.log(mesh_sizes.max()) + 0.25, 20, base=np.exp(1))
errs = np.exp(poly(np.log(dxs)))

fig, ax = plt.subplots()
ax.set_title("Convergence rate for ice shelf test case")
ax.set_xlabel("Mesh spacing (meters)")
ax.set_ylabel("$L^2$-norm relative error")
ax.set_xscale("log")
ax.set_yscale("log")
scatter_label = "relative errors for numerical solutions"
ax.scatter(mesh_sizes, errors, color="tab:orange", label=scatter_label)
plot_label = "log-log fit; error ~ $\delta x^2$"
ax.plot(dxs, errs, color="tab:blue", label=plot_label)
ax.legend()
fig.savefig(args.output)
