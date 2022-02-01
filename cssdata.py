# import from existing libraries
import pandas as pd
from matplotlib import pyplot as plt
import json

def csv_to_dataframe(input, exp_id, trial_id, num_headers=5):
    """
    convert csv to dataframe based on the specified exp_id and trial_id
    note that `input.json` data must be provided to input argument
    :param input: `input.json` data
    :param exp_id: experiment id
    :param trial_id: trial id
    :param num_headers: number of headers in csv file
    :return: dataframe
    """
    for experiment in input['experiments']:
        # iterate over experiments and find the experiment that corresponds to `exp_id` specified.
        if experiment['exp_id'] == exp_id:
            trials = experiment['trials']
            for trial in trials:
                # iterate over trials and fine the trial that corresponds to `trial_id` specified.
                if trial['id'] == trial_id:
                    file = trial['file']
                    relative_density = trial['relative_density']
                    # convert csv to dataframe
                    df = pd.read_csv(file, header=num_headers)
                    # add relative density to the first column of the dataframe.
                    df.insert(loc=0, column="Dr [%]", value=relative_density)
                    # compute and insert ru at the 5th column of the dataframes
                    # confining pressure value of the test is located at the first row of effective vertical stress column.
                    confining_pressure = df['Effective Vertical Stress [kPa]'][0]
                    ru = df['Excess Pore Pressure [kPa]']/confining_pressure
                    df.insert(loc=5, column="ru", value=ru)

    return df


def plot_trial(input, exp_id, trial_id, ncols=1, figsize=(13, 15)):
    # get data column names
    df = csv_to_dataframe(input=input, exp_id=exp_id, trial_id=trial_id)
    col_names = df.columns

    # plot for each data columns
    fig, axs = plt.subplots(nrows=len(col_names), ncols=ncols, figsize=figsize)
    axs_unroll = axs.flatten()
    for i, axi in enumerate(axs_unroll):
        axi.plot(df[col_names[i]])
        axi.set(xlabel='Data point')
        axi.set(ylabel=col_names[i])
    plt.savefig(f'./outputs/exp{exp_id}trial{trial_id}.png')
