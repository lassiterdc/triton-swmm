# use environment mrms_analysis
#%% import libraries and load directories
import xarray as xr
import pandas as pd
from  __filepaths import *
import rioxarray as rxr
import matplotlib.pyplot as plt
from glob import glob
import numpy as np
from pathlib import Path
from tqdm import tqdm

case_id = "norfolk_test"

#%% defining input filepaths
dem = fldr_triton_local + fldr_in_dem_asc + case_id + ".dem"

#%% defining output filepaths
lst_f_out_h = glob(fldr_out_bin2ascii + "H*")

lst_f_out_qxy = glob(fldr_out_bin2ascii + "Q*")

#%% inspecting inputs
ds_dem = rxr.open_rasterio(dem)

x = ds_dem.x.values
x.sort()
y = ds_dem.y.values
y.sort()

fig, ax = plt.subplots(dpi = 300)

ds_dem.plot(ax = ax, vmin = 0, vmax = 80)
ax.set_title("Input DEM for case {}".format(case_id))

plt.savefig("_scratch/" + '{}_dem.png'.format(case_id))
#%% inspecting outputs

# lst_fs = []
lst_tsteps = [] 
for f in lst_f_out_h:
    lst_tsteps.append(int(f.split("H_")[-1].split(".")[0].split("_")[0]))

df_files = pd.DataFrame(dict(timestep_min = lst_tsteps, files = lst_f_out_h))

df_files = df_files.sort_values("timestep_min")

lst_ds_h = []
for f in tqdm(df_files.files):
    df = pd.read_csv(f, sep = ' ', header = None)
    df.columns = x
    df = df.set_index(y)
    df = pd.melt(df, ignore_index=False, var_name = "x", value_name="H").reset_index(names = "y")
    df = df.set_index(["y", "x"])
    ds = df.to_xarray()
    lst_ds_h.append(ds)

ds_h = xr.concat(lst_ds_h, dim ="timestep")
# ds_h = ds_h.assign_coords(timestep = ds_h.timestep.values)
ds_h = ds_h.assign_coords(timestep = df_files.timestep_min.values)
ds_h.H.attrs["units"] ="ft"
ds_h.H.attrs["long_name"] ="water depth"

print("exporting netcdf...")
ds_h_loaded = ds_h.load()
ds_h_loaded.to_netcdf("{}_h.nc".format(case_id))
#%% generate animated visualization of ds_h
# https://climate-cms.org/posts/2019-09-03-python-animation.html
# ds_h = rxr.open_rasterio("{}_h.nc".format(case_id))
ds_h = xr.open_dataset("{}_h.nc".format(case_id))
from matplotlib import pyplot as plt, animation
from IPython.display import HTML, display

# test
fldr_plt = "_scratch/"
p = Path(fldr_plt)
p.mkdir(parents=True, exist_ok=True)

# for i in range(50,55):
#     ds_h.H.isel(dict(timestep = i)).plot(figsize=(12,6))
#     plt.savefig(fldr_plt + "{}_animation_frame{}".format(case_id, i), dpi = 300)
#     # plt.show()
#     plt.close()
# end test
#%% 
fig, ax = plt.subplots(figsize=(12,6))

cax = ds_h.H.isel(dict(timestep = 0)).plot(
    add_colorbar=True,
    cmap='coolwarm',
    vmin=np.floor(ds_h.H.min().values), vmax = np.ceil(ds_h.H.max().values),
    cbar_kwargs={
        'extend':'neither'
    }
)

def animate(frame):
    cax.set_array(ds_h.H.sel(dict(timestep = frame)).values.flatten())
    ax.set_title("timestep = " + str(frame))

ani = animation.FuncAnimation(
    fig,             # figure
    animate,         # name of the function above
    frames=ds_h.timestep.values.tolist(),       # Could also be iterable or list
    interval=200     # ms between frames
)

# HTML(ani.to_jshtml())

ani.save(fldr_plt + '{}_H_animation.mp4'.format(case_id))

#%% trying to understand the discontinuities
ds_h.H.isel(dict(x=300, y = 300)).plot()
plt.savefig(fldr_plt + "{}_inspecting_timeseries_for_single_cell.png".format(case_id))