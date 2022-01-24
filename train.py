# %%
import os
import json
import cssdata

# %% import data and dataframe

# open the `input.json` file
input = json.load(open("input.json", "r"))

# get dataframes for a trial
cssdata.csv_to_dataframe(input=input, exp_id=7, trial_id=3)

# # plot a trial
# cssdata.plot_trial(input=input, exp_id=7, trial_id=1)
