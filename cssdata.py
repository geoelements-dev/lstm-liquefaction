# import from existing libraries
import pandas as pd
from matplotlib import pyplot as plt

# import from my modules
from import_data import data_dirs


def csv_to_dataframe(basedir, experiments, drs):
    """
    make a list to contain dataframes for the whole data
    :param basedir: file directory where the raw data is saved
    :param experiments: a list of integer that specifies the experiment numbers that you are targeting
    :param drs: relative density of each trial
    :return: a list of dataframes that contains each `.csv` file
    """
    dfs = []  # make an empty list to contain dataframes

    # get the number of trials for each experiment
    for experiment in experiments:
        datadirs = data_dirs(experiment, basedir=basedir)  # get dir list of trial files in each exp
        num_experiments = len(experiments)  # get num of experiments
        num_trials = len(datadirs)  # get num of trials in each exp

        # make a list to contain a dataframes for each trial in the exp
        df_trial = []

        # get dataframes for a single trial and append it to `df_trial`
        for trial in range(num_trials):

            # make a dataframes for each trial
            df_single = pd.read_csv(datadirs[trial], header=5)  # make `.csv` to pandas `df`

            # insert Dr values at the first column of the dataframes
            df_single.insert(0, "Dr [%]", drs[experiment - 7][trial])

            # compute and insert ru at the 4th column of the dataframes
            # confining pressure of the test is the first (starting) effective vertical stress
            confining_pressure = df_single.iloc[0, 4]  # get conf pressure of the trial
            ru = df_single['Excess Pore Pressure [kPa]']/confining_pressure
            df_single.insert(5, "ru", ru)

            # Append this dataframes for the trial to `df_trial`
            df_trial.append(df_single)

        # Append `df_trial` to `dfs`
        dfs.append(df_trial)

    return dfs


def plot_trial(dataframes, expindex, trialindex):
    """plot all the columns in the dataframes at expindex and trialindex"""

    data_col_names = dataframes[expindex][trialindex].columns  # get data column names

    # plot for each data columns
    fig, axs = plt.subplots(nrows=len(data_col_names), ncols=1, figsize=(13, 15))
    axs_unroll = axs.flatten()
    for i, axi in enumerate(axs_unroll):
        axi.plot(dataframes[expindex][trialindex][data_col_names[i]])
        axi.set(xlabel='Data point')
        axi.set(ylabel=data_col_names[i])
    plt.show()


def print_experiment_summary(dataframes, timeindex, exp_num_list):
    """
    Some of data has a irregular time interval and different data points.
    This function look into those.
    """
    for k in range(len(exp_num_list)):
        len_data_list = len(dataframes[k])  # return num of trials at `k`th experiment
        print(f"Experiment{exp_num_list[k]}----------------------------------------")

        for i in range(len_data_list):
            print(f"*Trial-{i+1}")
            df = dataframes[k][i]  # load dataframes for the specified exp-trial
            index = df.index  # get the index of the dataframes
            index_last = index[-1]  # get the last index
            time = df['Time [sec]']  # get the time steps (sec)
            time_last = time[index_last]  # get the last time step
            time_choose = time[timeindex]  # get the time (sec) at the specified index
            time_interval = time_last/index_last  # get the time interval (sec) between each data point

            print(f"last index and time:{index_last}, {time_last}; interval: {time_interval:.4f}\n{timeindex} time step is: {time_choose} sec")


