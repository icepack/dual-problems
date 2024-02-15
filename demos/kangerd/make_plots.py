import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import rasterio
import firedrake
from firedrake import dx
import icepack

with firedrake.CheckpointFile("kangerdlugssuaq-year5.h5", "r") as chk:
    timesteps = np.array(chk.h5pyfile["timesteps"][:])
    mesh = chk.load_mesh()
    hs = [
        chk.load_function(mesh, name="thickness", idx=idx)
        for idx in range(len(timesteps))
    ]

# Make a plot of the volumes
volumes = np.array([firedrake.assemble(h * dx) for h in hs]) / 1e9
fig, axes = plt.subplots(figsize=(6.4, 3.2))
axes.set_title("Kangerdlugssuaq ice volume")
axes.set_xlabel("years")
axes.set_ylabel("ice volume (km${}^3$)")
axes.plot(timesteps, volumes)
fig.savefig("volumes.pdf", bbox_inches="tight")

# Fetch a satellite image
coords = mesh.coordinates.dat.data_ro
delta = 10e3
bbox = {
    "left": coords[:, 0].min() - delta,
    "right": coords[:, 0].max() + delta,
    "bottom": coords[:, 1].min() - delta,
    "top": coords[:, 1].max() + delta,
}
image_filename = icepack.datasets.fetch_mosaic_of_greenland()
with rasterio.open(image_filename, "r") as image_file:
    transform = image_file.transform
    window = rasterio.windows.from_bounds(**bbox, transform=transform)
    image = image_file.read(indexes=1, masked=True, window=window)

xmin, ymin, xmax, ymax = rasterio.windows.bounds(window, transform)
extent = (xmin, xmax, ymin, ymax)
imshow_kwargs = {"extent": extent, "cmap": "Greys_r", "vmin": 0e3, "vmax": 30e3}

# Make some contour plots of the thickness at the maximum and minimum extent
index_min = np.argmin(volumes)
index_max = np.argmax(volumes[index_min:]) + index_min
hmax = np.array([h.dat.data_ro.max() for h in hs]).max()
tmin = timesteps[index_min]
tmax = timesteps[index_max]

Q = firedrake.FunctionSpace(mesh, "CG", 1)
h_minvol = firedrake.Function(Q).project(hs[index_min])
h_maxvol = firedrake.Function(Q).project(hs[index_max])

fig, axes = plt.subplots()
axes.set_aspect("equal")
axes.set_xlabel("easting (m)")
axes.set_ylabel("northing")
axes.set_xlim((470e3, 510e3))
axes.set_ylim((-2308e3, -2270e3))
axes.ticklabel_format(axis="both", style="scientific", scilimits=(0, 0))
axes.imshow(image, **imshow_kwargs)
kwargs = {"levels": [9.0, 10.0], "axes": axes}
firedrake.tricontour(h_minvol, colors="tab:blue", **kwargs)
firedrake.tricontour(h_maxvol, colors="tab:green", **kwargs)
legend_elements = [
    Line2D([0], [0], color="tab:blue", lw=1, label=f"t = {tmin:.1f} yrs"),
    Line2D([0], [0], color="tab:green", lw=1, label=f"t = {tmax:.1f} yrs"),
]
axes.legend(handles=legend_elements, loc="upper right")
axes.set_title("Simulated terminus of Kangerdlugssuaq", pad=15)
fig.savefig("contours.pdf", bbox_inches="tight")
