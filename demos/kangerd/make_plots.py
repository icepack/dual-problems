import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib_scalebar.scalebar import ScaleBar
from matplotlib.lines import Line2D
import rasterio
import firedrake
from firedrake import dx
try:
    from firedrake.pyplot import tricontour
except ModuleNotFoundError:
    from firedrake import tricontour
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
inflection_indices = np.where(np.diff(np.sign(np.diff(volumes))))[0]
index_start, index_end = inflection_indices[1:3]
hmax = np.array([h.dat.data_ro.max() for h in hs]).max()
tmin = timesteps[index_start]
tmax = timesteps[index_end]

Q = firedrake.FunctionSpace(mesh, "CG", 1)

fig, axes = plt.subplots()
axes.set_aspect("equal")
axes.set_xlim((485e3, 510e3))
axes.set_ylim((-2308e3, -2285e3))
axes.get_xaxis().set_visible(False)
axes.get_yaxis().set_visible(False)
axes.imshow(image, **imshow_kwargs)
kwargs = {"levels": [9.0, 10.0], "axes": axes}
norm = matplotlib.colors.Normalize(vmin=tmin, vmax=tmax)
cmap = matplotlib.colormaps.get_cmap("viridis")
for index in range(index_start, index_end + 1):
    t = timesteps[index]
    color = matplotlib.colors.to_hex(cmap(norm(t)))
    tricontour(firedrake.Function(Q).project(hs[index]), colors=color, **kwargs)

scalebar = ScaleBar(1, units="m", length_fraction=0.4, location="lower right")
axes.add_artist(scalebar)

mappable = matplotlib.cm.ScalarMappable(norm=norm, cmap="viridis")
fig.colorbar(mappable, ax=axes, orientation="vertical", label="time (yrs)")
axes.set_title("Simulated terminus of Kangerdlugssuaq", pad=15)
fig.savefig("contours.pdf", bbox_inches="tight")
