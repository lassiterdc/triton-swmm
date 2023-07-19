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
fldr_plt = fldr_scratch = "_scratch/"
fldr_temp_netcdfs = fldr_scratch + "tmp_netcdfs/"
# %%

#%% defining input filepaths
dem = fldr_triton_local + fldr_in_dem_asc + case_id + ".dem"

# %%

#%% defining output filepaths
lst_f_out_h = glob(fldr_out_asc + "H*")

lst_f_out_mh = glob(fldr_out_asc + "MH*")

lst_f_out_qxy = glob(fldr_out_asc + "Q*")

# %% loading nodes
gdf_jxns = gpd.read_file(f_jxns).loc[:, ["NAME", "geometry"]]
gdf_strg = gpd.read_file(f_strg).loc[:, ["NAME", "geometry"]]
gdf_outfls = gpd.read_file(f_outfls).loc[:, ["NAME", "geometry"]]

gdf_nodes = pd.concat([gdf_jxns, gdf_strg, gdf_outfls])

gdf_subs = gpd.read_file(f_subs).loc[:, ["NAME", "geometry"]]


#%% loading subcatchments



#%% inspecting inputs
ds_dem = rxr.open_rasterio(dem)

x = ds_dem.x.values / meters_per_foot # converting back to coordinate system in feet
x.sort()
y = ds_dem.y.values / meters_per_foot # converting back to coordinate system in feet
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

df_files = pd.DataFrame(dict(tstep = lst_tsteps, files = lst_f_out_h))

df_files = df_files.sort_values("tstep")

p = Path(fldr_temp_netcdfs)
p.mkdir(parents=True, exist_ok=True)

# df_files = df_files.iloc[0:3, :]

# lst_ds_h = []
# for f in tqdm(df_files.files):
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
#%%
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

#%% generate animated visualization of ds_h
# https://climate-cms.org/posts/2019-09-03-python-animation.html
# ds_h = rxr.open_rasterio("{}_h.nc".format(case_id))
ds_h = xr.open_dataset("{}_h.nc".format(case_id), chunks = dict(timestep = "100MB"))
try:
    shutil.rmtree(fldr_temp_netcdfs)
except:
    pass
from matplotlib import pyplot as plt, animation
from IPython.display import HTML, display

# test

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
    zorder = 10,
    cbar_kwargs={
        'extend':'neither'
    }
)

gdf_subs.plot(ax=ax, edgecolor = "none", color = 'grey', alpha = 0.2, zorder = 1)

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
0
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
df_max_wlevel = df_max_wlevel.set_index(["y", "x"])
df_max_wlevel = df_max_wlevel.replace(0, np.nan)
ds_max_wlevel = df_max_wlevel.to_xarray()


ds_max_wlevel.max_wlevel_m.attrs["units"] ="m"
ds_max_wlevel.max_wlevel_m.attrs["long_name"] ="maximum water depth"

#%%
fig, ax = plt.subplots()

ds_max_wlevel.max_wlevel_m.plot(ax = ax, vmin=0, zorder = 10)

# gdf_nodes.plot(ax=ax, markersize = 5, zorder = 11, color = "none", edgecolor = "none")

gdf_subs.plot(ax=ax, edgecolor = "none", color = 'grey', alpha = 0.2, zorder = 1)

plt.savefig(fldr_plt + "{}_maximum_waterlevel.png".format(case_id))