import argparse
import pathlib
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
from matplotlib.animation import FuncAnimation
import rasterio
import firedrake
from firedrake import Constant, inner, assemble, dx
try:
    from firedrake.pyplot import tricontour, tripcolor
except ModuleNotFoundError:
    from firedrake import tricontour, tripcolor
import icepack

parser = argparse.ArgumentParser()
parser.add_argument("--with-movie", action="store_true")
args = parser.parse_args()

with firedrake.CheckpointFile("larsen-simulation.h5", "r") as chk:
    timesteps = chk.h5pyfile["timesteps"][:]
    time_to_calve = chk.h5pyfile.attrs["time_to_calve"]
    mesh = chk.load_mesh()
    hs = []
    for index in range(len(timesteps)):
        hs.append(chk.load_function(mesh, name="thickness", idx=index))

# Make a plot of the volumes
volumes = np.array([firedrake.assemble(h * dx) for h in hs]) / 1e9
fig, axes = plt.subplots()
axes.set_title("Larsen C ice volume")
axes.set_xlabel("year")
axes.set_ylabel("ice volume (km${}^3$)")
# TODO: add horizontal line at the time of the calving event
axes.plot(timesteps + 2015, volumes)
fig.savefig("volumes.pdf", bbox_inches="tight")

# Fetch a satellite image
coords = mesh.coordinates.dat.data_ro
delta = 50e3
bbox = {
    "left": coords[:, 0].min() - delta,
    "right": coords[:, 0].max() + delta,
    "bottom": coords[:, 1].min() - delta,
    "top": coords[:, 1].max() + delta,
}
image_filename = icepack.datasets.fetch_mosaic_of_antarctica()
with rasterio.open(image_filename, "r") as image_file:
    transform = image_file.transform
    window = rasterio.windows.from_bounds(**bbox, transform=transform)
    image = image_file.read(indexes=1, masked=True, window=window)

xmin, ymin, xmax, ymax = rasterio.windows.bounds(window, transform)
extent = (xmin, xmax, ymin, ymax)
imshow_kwargs = {
    "extent": extent,
    "cmap": "Greys_r",
    "vmin": 12e3,
    "vmax": 16.83e3,
    "interpolation": "none",
}

# Make a plot of the ice thickness before, during, and after the calving event
calving_index = np.argmin(np.abs(timesteps - time_to_calve)) + 1
time_indices = [0, calving_index, -1]

fig, axes = plt.subplots()
axes.set_aspect("equal")
axes.imshow(image, **imshow_kwargs)
axes.get_xaxis().set_visible(False)
axes.get_yaxis().set_visible(False)
colors = ["tab:blue", "tab:green", "tab:purple"]
handles = []
Q = firedrake.FunctionSpace(mesh, "CG", 1)
tricontour_kwargs = {"levels": [0.0, 20.0], "linewidths": 1.0}
for time_index, color in zip(time_indices, colors):
    h = firedrake.project(hs[time_index], Q)
    contourset = tricontour(
        h, colors=[(0, 0, 0, 0), color], axes=axes, **tricontour_kwargs
    )
    handles.append(contourset.legend_elements()[0][-1])
texts = [f"{2015 + timesteps[index]:.0f}" for index in time_indices]
axes.legend(handles, texts)
scalebar = AnchoredSizeBar(
    axes.transData,
    100e3,
    "100 km",
    "upper left",
    color="white",
    frameon=False,
    label_top=True,
)
axes.add_artist(scalebar)
fig.savefig("contours.pdf", bbox_inches="tight")


# Make a movie of the ice thickness
if args.with_movie:
    fn_plotter = firedrake.FunctionPlotter(mesh, num_sample_points=4)
    fig, axes = plt.subplots()
    axes.set_aspect("equal")
    axes.set_xlim((-2250e3, -2025e3))
    axes.set_ylim((1075e3, 1250e3))
    colors = tripcolor(hs[0], num_sample_points=4, axes=axes)
    fig.colorbar(colors)
    animation = FuncAnimation(
        fig, lambda h: colors.set_array(fn_plotter(h)), hs, interval=1e3/24
    )
    if not pathlib.Path("larsen-calving.mp4").exists():
        animation.save("larsen-calving.mp4")
