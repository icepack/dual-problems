import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors

parser = argparse.ArgumentParser()
parser.add_argument("--output", default="results.pdf")
args = parser.parse_args()

model_types = ["Ice shelf", "Ice stream"]
panel_letters = ["a", "b"]
fig, axes = plt.subplots(nrows=1, ncols=2, sharex=True, sharey=True)
axes[0].set_xlabel("Mesh spacing (m)")
axes[0].set_ylabel("$L^2$-norm relative error")
axes[1].get_yaxis().set_visible(False)

for ax, model_type, panel_letter in zip(axes, model_types, panel_letters):
    filename = model_type.lower().replace(" ", "_") + "_results.json"
    with open(filename, "r") as input_file:
        data = json.load(input_file)

    ax.set_title(f"{model_type} test case")
    ax.set_xscale("log")
    ax.set_xticks([2.5e2, 5e2, 1e3], labels=["$250$", "$500$", "$1000$"])
    ax.set_yscale("log")
    kw = {"xy": (0.05, 0.92), "xycoords": "axes fraction", "fontsize": 16}
    ax.annotate(panel_letter, **kw)

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
