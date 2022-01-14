# %%
import os
import json

import cssdata

# %% import data and dataframe

cwd = os.getcwd()  # get current working directory
data_dir = os.path.join(cwd, "rawdata")  # define the path for the rawdata
inputfile = "input.json"  # name of the `input.json`
inputfile_dir = os.path.join(cwd, inputfile)  # define the path for the `input.json`

# open the `input.json` file
with open(inputfile_dir, "r") as f:
    json_data = json.load(f)

experiments = json_data["exp_num_list"]  # experiment list that you are importing
dr_exp7 = json_data["dr_exp7"]  # relative density of each trial in experiment 7
dr_exp8 = json_data["dr_exp8"]  # relative density of each trial in experiment 8
dr_exp9 = json_data["dr_exp9"]  # relative density of each trial in experiment 9
dr_exp10 = json_data["dr_exp10"]  # relative density of each trial in experiment 10
drs = [dr_exp7, dr_exp8, dr_exp9, dr_exp10]  # collect relative densities in a list

# get dataframes for all trials
dfs = cssdata.csv_to_dataframe(basedir=data_dir, experiments=experiments, drs=drs)

