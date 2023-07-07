#%% key user defined inputs
triton_model_name = "norfolk_test"
num_srcs = 185 # corresponds to the num of nodes with flooding in the SWMM model
constant_man_bool = True
const_man = 0.035
num_ext_bc = 0
tstep_s = .01
reporting_tstep_s = 60*5
sim_dur_s = 30 * 60 * 60 # 30 hours
#%% TRITON
fldr_ornl_local = "D:/Dropbox/_GradSchool/_ORNL_internship/"
fldr_triton_local = fldr_ornl_local + "triton-swmm/"

fldr_ornl_hpc = "/project/quinnlab/dcl3nd/TRITON/"
fldr_triton_hpc = fldr_ornl_hpc + "triton/"

# input paths
fldr_inputs = "input/"
fldr_cfg = "input/cfg/"
fldr_in_dem = fldr_inputs + "dem/"
fldr_in_extbc = fldr_inputs + "extbc/"
fldr_in_mann = fldr_inputs + "mann/"
fldr_in_stageloc = fldr_inputs + "stageloc/"
fldr_in_strmflow = fldr_inputs + "strmflow/"
fldr_in_dem_asc = fldr_in_dem + "asc/"

f_template_cfg_local = fldr_triton_local + fldr_cfg + "_input_template.cfg"

f_cfg = fldr_triton_local + fldr_cfg  + triton_model_name + ".cfg"
# lst_keys = ["DEM", "HYDROGRAPH", "HYDO_SRC_LOC", "MANNINGS", "CONST_MAN_BOOL", "CONST_MAN",
#             "NUM_EXT_BC", "EXTBC_DIR", "EXTBC_FILE", "SIM_DUR_S", "TSTEP_S", "REPORTING_TSTEP_S"]

f_in_dem = fldr_in_dem_asc + triton_model_name + ".dem"
f_in_hydrograph = fldr_in_strmflow + triton_model_name + ".hyg"
f_in_hydro_src_loc = fldr_in_strmflow + triton_model_name + ".txt"
f_in_mannings = fldr_in_mann + triton_model_name + ".mann"
f_in_extbc_file = fldr_in_extbc + triton_model_name + ".extbc"

if constant_man_bool == True:
    constant_man_bool_tmplt = ""
    mann_file_toggle = "# "
else:
    constant_man_bool_tmplt = "# "
    mann_file_toggle = ""


d_input = dict(DEM = f_in_dem, NUM_SOURCES = num_srcs, HYDROGRAPH = f_in_hydrograph, HYDO_SRC_LOC = f_in_hydro_src_loc,
               MANNINGS = f_in_mannings, CONST_MAN_BOOL = constant_man_bool_tmplt, 
               MAN_FILE_TOGGLE = mann_file_toggle,
               CONST_MAN = const_man, NUM_EXT_BC = num_ext_bc, EXTBC_DIR = fldr_in_extbc,
               EXTBC_FILE = f_in_extbc_file, SIM_DUR_S = sim_dur_s, TSTEP_S = tstep_s, 
               REPORTING_TSTEP_S = reporting_tstep_s)

# output paths
fldr_outputs = fldr_triton_local + "output/"
fldr_out_asc = fldr_outputs + "asc/"
fldr_out_bin2ascii = fldr_outputs + "bin2ascii/"

#%% additional data
# SWMM
# fldr_repo_stormy = "D:/Dropbox/_GradSchool/_norfolk/stormy/"
fldr_triton_local_data = fldr_triton_local + "_data/"
fldr_swmm = fldr_triton_local_data + "swmm/"
model_name = "hague_V1_using_norfolk_data"
f_inp = fldr_swmm + model_name + ".inp"
f_out = fldr_swmm + model_name + ".out"

# swmm entities 
fldr_shps = fldr_swmm + "exported_layers/"
f_jxns = fldr_shps + "junctions.shp"
f_strg = fldr_shps + "storages.shp"
f_outfls = fldr_shps + "outfalls.shp"

# DEM
# fldr_triton_hpc_data = fldr_triton_hpc + "_data/"
f_dem_raw = fldr_triton_local_data + "dem_with_buildings.tif"
f_dem_processed = fldr_triton_local + "input/dem/asc/" + triton_model_name + ".dem"
#%% constants
meters_per_foot = 0.3048
square_meters_per_square_foot = meters_per_foot * meters_per_foot
cubic_meters_per_cubic_foot = meters_per_foot*meters_per_foot*meters_per_foot