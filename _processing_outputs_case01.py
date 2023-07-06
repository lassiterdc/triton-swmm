#%% import libraries
import xarray as xr
import pandas as pd
from glob import glob

#%% define filepaths
fldr_outs = "D:/Dropbox/_GradSchool/_ORNL internship/triton-master/output/"
fldr_asc = fldr_outs + "asc/"
fldr_bin = fldr_outs + "bin/"
fldr_cfg = fldr_outs + "cfg/"
fldr_series = fldr_outs + "series/"
f_perf = fldr_outs + "performance.txt"

#%%
df_perf = pd.read_csv(f_perf)

asc_files = glob(fldr_asc+"*.out")
bin_files = glob(fldr_bin+"*.out")

for f in asc_files:
    df = pd.read_table(f)