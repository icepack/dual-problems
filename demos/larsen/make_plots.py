import argparse
import pathlib
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
from matplotlib.animation import FuncAnimation
import rasterio
import firedrake
from firedrake import Constant, inner, assemble, dx
import icepack

parser = argparse.ArgumentParser()
parser.add_argument("--with-movie", action="store_true")
args = parser.parse_args()

with firedrake.CheckpointFile("larsen-simulation.h5", "r") as chk:
    timesteps = chk.h5pyfile["timesteps"][:]
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
imshow_kwargs = {"extent": extent, "cmap": "Greys_r", "vmin": 12e3, "vmax": 16.38e3}

# Make a plot of the ice thickness before, during, and after the calving event
fig, axes = plt.subplots(nrows=1, ncols=3, sharex=True, sharey=True)
for ax, text in zip(axes, ["2015", "2019", "2023"]):
    ax.set_title(text)
    ax.set_aspect("equal")
    ax.imshow(image, **imshow_kwargs)
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    ax.set_xlim((-2250e3, -2025e3))
    ax.set_ylim((1075e3, 1250e3))
time_indices = [np.argmin(np.abs(timesteps - time)) for time in [0.0, 4.5, 12.0]]
for index, time_index in enumerate(time_indices):
    firedrake.tripcolor(hs[time_index], axes=axes[index])
fig.savefig("thickness.pdf", bbox_inches="tight")

fig, axes = plt.subplots()
axes.set_aspect("equal")
axes.imshow(image, **imshow_kwargs)
axes.get_xaxis().set_visible(False)
axes.get_yaxis().set_visible(False)
colors = ["tab:blue", "tab:green", "tab:purple"]
handles = []
for time_index, color in zip(time_indices, colors):
    contourset = firedrake.tricontour(
        hs[time_index], levels=[0.0, 20.0], colors=["black", color], axes=axes,
    )
    handles.append(contourset.legend_elements()[0][-1])
texts = ["2015", "2019", "2027"]
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
    colors = firedrake.tripcolor(hs[0], num_sample_points=4, axes=axes)
    fig.colorbar(colors)
    animation = FuncAnimation(
        fig, lambda h: colors.set_array(fn_plotter(h)), hs, interval=1e3/24
    )
    if not pathlib.Path("larsen-calving.mp4").exists():
        animation.save("larsen-calving.mp4")
