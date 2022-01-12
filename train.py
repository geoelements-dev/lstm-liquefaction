# %%
import tensorflow as tf
import numpy as np
import os
import json
from matplotlib import pyplot as plt

from tensorflow.keras.layers import Dense, Dropout, SimpleRNN, LSTM, GRU, Flatten
from tensorflow.keras import optimizers
from tensorflow.keras.callbacks import EarlyStopping

import cssdata

# %% import data and dataframe

cwd = os.getcwd()  # get current working directory
data_dir = os.path.join(cwd, "rawdata")  # define the path for the rawdata

exp_num_list = cssdata.exp_num_list()  # get exp_num_list that you want to consider (by default, [7, 8, 9, 10])
drs = cssdata.relative_density()  # get relative density (Dr) data for each trial

# get dataframe for all trials from experiment 7, 8, 8, and 10
df_all = cssdata.to_dataframe(data_dir=data_dir, exp_num_list=exp_num_list, drs=drs)

