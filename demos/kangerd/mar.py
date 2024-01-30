import argparse
import pathlib
import subprocess
import numpy as np
import xarray
import firedrake
import icepack


def fetch(year):
    base_url = "ftp://ftp.climato.be/fettweis/MARv3.12/Greenland/PROTECT/ERA5"
    filename = f"MARv3.12-yearly-ERA5-{year}.nc"
    if not pathlib.Path(filename).is_file():
        subprocess.run(f"curl -O {base_url}/{filename}".split())
    return xarray.open_dataset(filename, decode_times=False)


parser = argparse.ArgumentParser()
parser.add_argument("--start-year", type=int, default=2006)
parser.add_argument("--end-year", type=int, default=2021)
args = parser.parse_args()

mesh = firedrake.Mesh("kangerdlugssuaq-enlarged.msh")
coords = mesh.coordinates.dat.data_ro
x, y = coords[:, 0], coords[:, 1]
delta = 5e3
xrange = slice(x.min() - delta, x.max() + delta)
yrange = slice(y.min() - delta, y.max() + delta)

years = range(args.start_year, args.end_year + 1)
smb = xarray.concat(
    [fetch(year).sel(x=xrange, y=yrange)["SMB"] for year in years], dim="time"
)

bedmachine_filename = icepack.datasets.fetch_bedmachine_greenland()
bedmachine = xarray.open_dataset(bedmachine_filename)
surface = bedmachine["surface"].interp_like(smb).expand_dims(dim={"time": smb["time"]})

μ_smb, σ_smb = float(smb.mean()), float(smb.std())
μ_surf, σ_surf = float(surface.mean()), float(surface.std())
corr = np.tensordot((smb - μ_smb) / σ_smb, (surface - μ_surf) / σ_surf, axes=3) / smb.size
lapse_rate = corr * σ_smb / σ_surf
baseline = μ_smb - lapse_rate * μ_surf
print(f"SMB ~= {lapse_rate:0.2f} * s {baseline:+0.2f} mmWe/yr; r² = {corr:.2f}")
