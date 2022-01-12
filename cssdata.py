# import from existing libraries
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

# import from my modules
from dataimportingfunctions import get_data_dir_list

def exp_num_list():
    """"
    define experiment numbers that you want to include from the whole data
    """

    exp_num_list = [7, 8, 9, 10]
    return exp_num_list


def relative_density():
    """
    define relative densities for each experiment and trials
    """

    dr_exp7 = np.array([55, 54, 47, 56, 52, 52, 39, 52, 41, 43,
                        39, 39, 33, 35, 43, 35, 41, 44, 53]) / 100
    dr_exp8 = np.array([76, 94, 89, 77, 72, 73, 85, 88, 90, 74]) / 100
    dr_exp9 = np.array([41, 47, 47, 40, 39, 43, 41, 49, 41, 37,
                        44, 47, 48, 44, 50, 47, 51, 48, 50, 50, 40, 44,
                        44, 45, 41, 44, 43, 41, 43, 37, 42, 43, 36, 50,
                        42, 45, 49, 43, 39, 43, 47, 48, 55, 44, 49, 41,
                        55]) / 100
    dr_exp10 = np.array([78, 72, 84, 79, 82, 86, 89, 74, 70, 74, 80, 78, 89,
                         84, 89, 84, 81, 83, 91, 86, 78, 73, 72, 71, 76, 80, 73, 74, 73]) / 100
    drs = [dr_exp7, dr_exp8, dr_exp9, dr_exp10]

    return drs


def to_dataframe(data_dir, exp_num_list=exp_num_list(), drs=relative_density()):
    """make a list to contain dataframe for the whole data"""

    df_all = []  # make an empty list to contain dataframes

    # get the number of trials for each experiment
    for exp in exp_num_list:
        data_dir_list = get_data_dir_list(exp, basedir=data_dir)  # get dir list of trial files in each exp
        num_exps = len(exp_num_list)  # get num of experiments
        num_trials = len(data_dir_list)  # get num of trials in each exp

        # make a list to contain a dataframe for each trial in the exp
        df_trial = []

        # get dataframe for a single trial and append it to `df_trial`
        for trial in range(num_trials):

            # make a dataframe for each trial
            data_dir_trial = data_dir_list[trial]  # get directory of each `trial.csv`
            df_single = pd.read_csv(data_dir_trial, header=5)  # make `.csv` to pandas `df`

            # insert Dr values at the first column of the dataframe
            df_single.insert(0, "Dr [%]", drs[exp - 7][trial])

            # compute and insert ru at the 4th column of the dataframe
            # confining pressure of the test is the first (starting) effective vertical stress
            confining_pressure = df_single.iloc[0, 4]  # get conf pressure of the trial
            ru = df_single['Excess Pore Pressure [kPa]']/confining_pressure
            df_single.insert(5, "ru", ru)

            # Append this dataframe for the trial to `df_trial`
            df_trial.append(df_single)

        # Append `df_trial` to `df_all`
        df_all.append(df_trial)

    return df_all


def plot_trial(dataframe, expindex, trialindex):
    """plot all the columns in the dataframe at expindex and trialindex"""

    data_col_names = dataframe[expindex][trialindex].columns  # get data column names

    # plot for each data columns
    fig, axs = plt.subplots(nrows=len(data_col_names), ncols=1, figsize=(13, 15))
    axs_unroll = axs.flatten()
    for i, axi in enumerate(axs_unroll):
        axi.plot(dataframe[expindex][trialindex][data_col_names[i]])
        axi.set(xlabel='Data point')
        axi.set(ylabel=data_col_names[i])


def look_into_data(dataframe, timeindex, exp_num_list=exp_num_list()):
    """
    Some of data has a irregular time interval and different data points.
    This function look into those.
    """
    for k in range(len(exp_num_list)):
        len_data_list = len(dataframe[k])  # return num of trials at `k`th experiment
        print(f"Experiment{exp_num_list[k]}----------------------------------------")

        for i in range(len_data_list):
            print(f"*Trial-{i+1}")
            df = dataframe[k][i]  # load dataframe for the specified exp-trial
            index = df.index  # get the index of the dataframe
            index_last = index[-1]  # get the last index
            time = df['Time [sec]']  # get the time steps (sec)
            time_last = time[index_last]  # get the last time step
            time_choose = time[timeindex]  # get the time (sec) at the specified index
            time_interval = time_last/index_last  # get the time interval (sec) between each data point

            print(f"last index and time:{index_last}, {time_last}; interval: {time_interval:.4f}\n{timeindex} time step is: {time_choose} sec")


