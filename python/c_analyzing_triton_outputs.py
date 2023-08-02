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
import geopandas as gpd
from tqdm import tqdm
import shutil

case_id = "norfolk_test"

fldr_temp_netcdfs = fldr_scratch + "tmp_netcdfs/"
#%% defining functions for reading binary outputs
HEADER = 2

def read_binary_matrix(filepath):
    with open(filepath, "rb") as file:
        dim_arr = np.fromfile(file, dtype=np.float64, count=HEADER)
        nrows, ncols = dim_arr.astype(np.int32)

        arr = np.fromfile(file, dtype=np.float64, count=nrows * ncols)

    return nrows, ncols, arr.reshape(nrows, ncols)

def plot_matrix(matrix):
    plt.imshow(matrix, cmap="viridis", interpolation="nearest")
    plt.colorbar()
    plt.title("Matrix Visualization")
    plt.xlabel("Columns")
    plt.ylabel("Rows")
    plt.show()

#%% defining input filepaths
dem = fldr_triton_local + fldr_in_dem_asc + case_id + ".dem"

# %%

#%% loading data
# defining output filepaths
lst_f_out_h = glob(fldr_outputs + "H*")

lst_f_out_mh = glob(fldr_outputs + "MH*")

# loading nodes
gdf_nodes = gpd.read_file(f_nodes).loc[:, ["NAME", "geometry"]]

gdf_wshed = gpd.read_file(f_watershed)

# loading dem
ds_dem = rxr.open_rasterio(dem)

x = ds_dem.x.values
# x.sort()
y = ds_dem.y.values
# y.sort()

#%% inspecting DEM
fig, ax = plt.subplots(dpi = 300)

ds_dem.plot(ax = ax, vmin = 0, vmax = 80)
ax.set_title("Input DEM for case {}".format(case_id))

plt.savefig(fldr_plt + '{}_dem.png'.format(case_id))
#%% writing outputs to netcdf
lst_tsteps = [] 
for f in lst_f_out_h:
    lst_tsteps.append(int(f.split("H_")[-1].split(".")[0].split("_")[0]))

df_files = pd.DataFrame(dict(tstep = lst_tsteps, files = lst_f_out_h))

df_files = df_files.sort_values("tstep")

p = Path(fldr_temp_netcdfs)
p.mkdir(parents=True, exist_ok=True)

# df_files = df_files.iloc[0:3, :]

# lst_ds_h = []
# for f in tqdm(df_files.files):
if output_type == "bin":
    for ind, row in tqdm(df_files.iterrows()):
        f = row.files
        tstep = row.tstep
        with open(f, mode='rb') as file: # b is important -> binary
            fileContent = file.read()
            # print(fileContent)
if output_type == "asc":
    for ind, row in tqdm(df_files.iterrows()):
        f = row.files
        tstep = row.tstep
        df = pd.read_csv(f, sep = ' ', header = None)
        df.columns = x
        df = df.set_index(y)
        df = pd.melt(df, ignore_index=False, var_name = "x", value_name="H").reset_index(names = "y")
        df["timestep"] = tstep
        df = df.set_index(["y", "x", "timestep"])
        df = df.replace(0, np.nan) # replace 0 with nan so it doesn't appear in the plots
        ds = df.to_xarray()
        ds.to_netcdf(fldr_temp_netcdfs + "{}_h.nc".format(tstep), encoding= {"H":{"zlib":True}})

    # lst_ds_h.append(ds)

# ds_h = xr.concat(lst_ds_h, dim ="timestep")
ds_h = xr.open_mfdataset(fldr_temp_netcdfs + "{}_h.nc".format("*"))
# ds_h = ds_h.assign_coords(timestep = ds_h.timestep.values)
# ds_h = ds_h.assign_coords(timestep = df_files.timestep_min.values)
ds_h.H.attrs["units"] ="m"
ds_h.H.attrs["long_name"] ="water depth"

print("exporting netcdf...")
# ds_h_loaded = ds_h.load()
# ds_h_loaded.to_netcdf("{}_h.nc".format(case_id))
ds_h.to_netcdf("{}_h.nc".format(case_id), encoding= {"H":{"zlib":True}})

try:
    shutil.rmtree(fldr_temp_netcdfs)
except:
    pass

#%% plotting maximum water level
lst_tsteps = [] 
for f in lst_f_out_mh:
    lst_tsteps.append(int(f.split("H_")[-1].split(".")[0].split("_")[0]))

df_files_max_wlevel = pd.DataFrame(dict(timestep_min = lst_tsteps, files = lst_f_out_mh))

f_max_wlevel = df_files_max_wlevel.files[df_files_max_wlevel.timestep_min.idxmax()]

df_max_wlevel = pd.read_csv(f_max_wlevel, sep = ' ', header = None)
df_max_wlevel.columns = x
df_max_wlevel = df_max_wlevel.set_index(y)
df_max_wlevel = pd.melt(df_max_wlevel, ignore_index=False, var_name = "x", value_name="max_wlevel_m").reset_index(names = "y")
df_max_wlevel = df_max_wlevel.set_index(["x", "y"])
df_max_wlevel = df_max_wlevel.replace(0, np.nan)
ds_max_wlevel = df_max_wlevel.to_xarray()


ds_max_wlevel.max_wlevel_m.attrs["units"] ="m"
ds_max_wlevel.max_wlevel_m.attrs["long_name"] ="maximum water depth"

fig, ax = plt.subplots()

ds_max_wlevel.max_wlevel_m.plot(x="x", y = "y", ax = ax, vmin=0, zorder = 10)

# gdf_nodes.plot(ax=ax, markersize = 5, zorder = 11, color = "none", edgecolor = "none")

gdf_wshed.plot(ax=ax, edgecolor = "black", color = 'none', alpha = 0.2, zorder = 1)

plt.savefig(fldr_plt + "{}_maximum_waterlevel.png".format(case_id))
#%% generate animated visualization of ds_h
# https://climate-cms.org/posts/2019-09-03-python-animation.html
# ds_h = rxr.open_rasterio("{}_h.nc".format(case_id))
import time

start_time = time.time()
# print("opening netcdf of outputs")
ds_h = xr.open_dataset("{}_h.nc".format(case_id), chunks = dict(timestep = 1))
# print("time: {}".format(round(time.time() - start_time, 2)))

from matplotlib import pyplot as plt, animation

p = Path(fldr_plt)
p.mkdir(parents=True, exist_ok=True)

fig, ax = plt.subplots(figsize=(12,6))

# print("computing colorbar limits")
vmax = df_max_wlevel.max().values
vmin = 0

# print("creating cax...")
first_tstep_loaded = ds_h.H.isel(dict(timestep = 0)).load()
cax = first_tstep_loaded.plot(
    add_colorbar=True,
    cmap='coolwarm',
    vmin = vmin, vmax = vmax,
    zorder = 10,
    cbar_kwargs={
        'extend':'neither'
    }
)
# print("time: {}".format(round(time.time() - start_time, 2)))

# print("plotting watershed")
gdf_wshed.plot(ax=ax, edgecolor = "none", color = 'grey', alpha = 0.2, zorder = 1)
print("time: {}".format(round(time.time() - start_time, 2)))
def animate(frame):
    cax.set_array(ds_h.H.sel(dict(timestep = frame)).values.flatten())
    ax.set_title("timestep = " + str(frame))
# print("time: {}".format(round(time.time() - start_time, 2)))
# print('creating the animation...')
ani = animation.FuncAnimation(
    fig,             # figure
    animate,         # name of the function above
    frames=ds_h.timestep.values.tolist(),       # Could also be iterable or list
    interval=200     # ms between frames
)
# print("time: {}".format(round(time.time() - start_time, 2)))
# HTML(ani.to_jshtml())
print('exporting the animation. This can take around 45 minutes.')
ani.save(fldr_plt + '{}_H_animation.gif'.format(case_id), writer = animation.PillowWriter(fps=5)) #, writer=animation.FFMpegWriter(fps=8))
print("time: {}".format(round(time.time() - start_time, 2)))
