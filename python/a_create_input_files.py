#%% import libraries
import xarray as xr
import rioxarray as rxr
import pandas as pd
import numpy as np
import geopandas as gpd
import rasterio as rio
from rasterio.plot import plotting_extent
import seaborn as sns
import earthpy as et
import earthpy.plot as ep
import matplotlib.pyplot as plt
from pathlib import Path
from pyswmm import Output
from swmm.toolkit.shared_enum import NodeAttribute

sns.set(font_scale=1.5, style="white")

from __filepaths import *

#%% load data
gdf_nodes = gpd.read_file(f_nodes).loc[:, ["NAME", "geometry"]]
gdf_bc = gpd.read_file(f_shp_bndry_cond)
rds_dem = rxr.open_rasterio(f_dem_raw)

# fill gaps in dem
if (rds_dem.values<-100).sum() > 0:
    rds_dem = rds_dem.rio.interpolate_na(method = "nearest")
#%% working with dem
with rio.open(f_dem_raw) as dem_src:
    lidar_dem = dem_src.read(1, masked=False) 
    lidar_dem_meta = dem_src.profile

fig, ax = plt.subplots()

rds_dem.plot(ax = ax, vmin = rds_dem.min().values, vmax = rds_dem.max().values)

xlim = ax.get_xlim()
ylim = ax.get_ylim()

gdf_nodes.plot(ax = ax)

ax.set_xlim(xlim)
ax.set_ylim(ylim)

plt.show()

ep.hist(lidar_dem,
        figsize=(10, 6),
        title="Histogram of Elevation")
plt.show()


#%% create dem file
fillna_val = -9999

# df_dem = df_dem.fillna(fillna_val)

cellsize_x = pd.DataFrame(np.unique(np.diff(rds_dem.x.values), return_counts = True)).T
cellsize_y = pd.DataFrame(np.unique(np.diff(rds_dem.y.values), return_counts = True)).T

cellsize_x.columns = ["diff", "count"]

cellsize = cellsize_x["diff"][cellsize_x["count"].idxmax()]

input_dem_metadata = {"ncols         ":rds_dem.x.shape[0],
                          "nrows         ":rds_dem.y.shape[0], 
                          "xllcorner     ":rds_dem.x.values.min() - cellsize/2, # adjusted from center to corner
                          "yllcorner     ":rds_dem.y.values.min() - cellsize/2, # adjusted from center to corner
                          "cellsize      ":cellsize,
                          "NODATA_value  ":fillna_val}

 # np.unique(np.diff(rds_dem.x.values), return_counts = True)

dem_file_path = Path(f_dem_processed)
dem_file_path.parent.mkdir(parents=True, exist_ok=True)

f = open(f_dem_processed, "w")
for key in input_dem_metadata:
    f.write(key + str(input_dem_metadata[key]) + "\n")
f.close()


# create dataframe with the right shape
df_dem_long = rds_dem.to_dataframe("elevation").reset_index().loc[:, ["x", "y", "elevation"]]
df_dem = df_dem_long.pivot(index = "y", columns = "x", values = "elevation")

# ensure the x and y values are ordered properly
x_ordered = df_dem_long.x.sort_values().unique()
y_ordered = df_dem_long.y.sort_values().unique()

df_dem = df_dem.loc[y_ordered, x_ordered]

# pad with zeros to achieve consistent spacing in the resulting file
target_decimal_places = 5
longest_num = len(str(df_dem.abs().max().max()).split(".")[0]) + target_decimal_places + 1

def flt_to_str_certain_num_of_characters(flt):
    flt = round(flt, target_decimal_places)
    str_flt = flt.astype(str)
    str_flt = str_flt.apply(lambda x: x.ljust(longest_num, '0'))
    return str_flt

df_dem_padded = df_dem.apply(flt_to_str_certain_num_of_characters)

df_dem_padded.to_csv(f_dem_processed, mode = "a", index = False, header=False, sep=" ", float_format = "{:10.4f}")

#%% create hydrograph time series files
flding_first_line = "%Norfolk Storm Sewer Flooding\n"
flding_second_line = "%Time(hr) Discharge(cms)\n"

wlevel_first_line = "%Norfolk Water Level Boundary Condition\n"
wlevel_second_line = "%Time(hr) water_elevation (m)\n"

lst_nodes_with_flooding = []

d_flooding_time_series = dict()
d_wlevel_time_series = dict()
count = -1
with Output(f_out) as out:
    for key in out.nodes:
        count += 1
        d_t_series = pd.Series(out.node_series(key, NodeAttribute.FLOODING_LOSSES)) #cfs
        if count == 0:
            tstep_seconds = float(pd.Series(d_t_series.index).diff().mode().dt.seconds)
            tseries = pd.Series(d_t_series.index).diff().dt.seconds / 60 / 60
            tseries.loc[0] = 0
            d_flooding_time_series["time_hr"] = tseries.cumsum().values

        # convert from cfs to cms
        d_t_series = d_t_series * cubic_meters_per_cubic_foot
        if d_t_series.sum() > 0:
            lst_nodes_with_flooding.append(key)
            d_flooding_time_series[key] = d_t_series.values

        if key == "E147007": #outfall node
            d_t_series = pd.Series(out.node_series(key, NodeAttribute.HYDRAULIC_HEAD)) # feet
            tstep_seconds = float(pd.Series(d_t_series.index).diff().mode().dt.seconds)
            tseries = pd.Series(d_t_series.index).diff().dt.seconds / 60 / 60
            tseries.loc[0] = 0
            d_wlevel_time_series["time_hr"] = tseries.cumsum().values
            d_wlevel_time_series["water_level_m"] = (d_t_series * meters_per_foot).values
        # sum all flooded volumes and append lists
        # lst_tot_node_flding.append(d_t_series.sum())
        

df_node_flooding = pd.DataFrame(d_flooding_time_series)
f = open(fldr_triton_local + f_in_hydrograph, "w")
f.write(flding_first_line + flding_second_line)
f.close()
df_node_flooding.to_csv(fldr_triton_local + f_in_hydrograph, mode = "a", index = False, header=False)

df_waterlevel = pd.DataFrame(d_wlevel_time_series)
f = open(fldr_triton_local + f_in_extbc_wlevel, "w")
f.write(wlevel_first_line + wlevel_second_line)
f.close()
df_waterlevel.to_csv(fldr_triton_local + f_in_extbc_wlevel, mode = "a", index = False, header=False)
#%% create hydrograph files
# ensure the xy locations ROW aligns with the hydrograph COLUMN
hydrograph_col_order = df_node_flooding.columns[1:]

gdf_nodes = gdf_nodes.set_index("NAME").loc[hydrograph_col_order].reset_index(names="NAME")

str_first_line = "%X-Location,Y-Location"

f = open(fldr_triton_local + f_in_hydro_src_loc, "w")
f.write(str_first_line + "\n")

for geom in gdf_nodes.geometry:
    x = geom.x
    y = geom.y
    f.write("{},{}\n".format(x, y))
f.close()

# verifying that all nodes are within the DEM
xllcorner = rds_dem.x.values.min()
yllcorner = rds_dem.y.values.min()

df_xylocs = pd.read_csv(fldr_triton_local + f_in_hydro_src_loc, header=0, names = ["x", "y"])

# print("min x node: {}, min x DEM: {}".format(df_xylocs.x.min(), xllcorner))
# print("min y node: {}, min y DEM: {}".format(df_xylocs.y.min(), yllcorner))

if df_xylocs.x.min() < xllcorner:
    print("problem with x's")

if df_xylocs.y.min() < yllcorner:
    print("problem with y's")

#%% create external boundary condition file
str_line1 = "% BC Type, X1, Y1, X2, Y2, BC"

def extract_vertex_coordinates(geometry):
    # Ensure the geometry is a LineString or MultiLineString
    if geometry.geom_type in ['LineString', 'MultiLineString']:
        return list(geometry.coords)
    else:
        return None
    
gdf_bc['vertices'] = gdf_bc['geometry'].apply(extract_vertex_coordinates)

lst_x = []
lst_y = []
for geom in gdf_bc.vertices:
    for vertex in geom:
        # print(vertex)
        lst_x.append(vertex[0])
        lst_y.append(vertex[1])

# find x and ys at edge of DEM representing the boundary condition
min_x = min(lst_x)
min_y = min(lst_y)
max_x = max(lst_x)
max_y = max(lst_y)

def find_closest_dem_coord(x_val, y_val, BC_side):
    if BC_side == "left":
        dem_xs = rds_dem.x.values - cellsize/2
        x_coord = min(dem_xs)

        dem_ys = rds_dem.y.values
        y_coord = dem_ys[np.argmin(np.abs(dem_ys - y_val))]
    else:
        import sys
        sys.exit("boundary condition location not defined")
    return x_coord, y_coord

x1, y1 = find_closest_dem_coord(min_x, min_y, BC_side)
x2, y2 = find_closest_dem_coord(max_x, max_y, BC_side)

str_line2 = "{},{},{},{},{},{}".format(BC_type, x1, y1, x2, y2, BC)

# write file
f = open(fldr_triton_local + f_in_extbc_file, "w")
f.write(str_line1 + "\n")
f.write(str_line2 + "\n")
f.close()