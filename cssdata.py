import numpy as np
import pandas as pd
import os
from matplotlib import pyplot as plt

def augment_data(df, add_length=900, time_step=0.012):

    # add static state CSS test result (before test starts) to augment data
    css_start_time = time_step*add_length
    add_Dr = np.full((add_length, 1), df["Dr [%]"][0])
    add_Time = np.reshape(np.linspace(0, css_start_time, add_length), (add_length, 1))
    add_ShearStrain = np.full((add_length, 1), 0)
    add_SheerStress = np.full((add_length, 1), 0)
    add_ConfPressure = np.full((add_length, 1), df["Effective Vertical Stress [kPa]"][0])
    add_ru = np.full((add_length, 1), df["ru"][0])
    add_PWP = np.full((add_length, 1), df["Excess Pore Pressure [kPa]"][0])
    # aggregate as array
    add_data = np.hstack(
        (add_Dr, add_Time, add_ShearStrain, add_SheerStress, add_ConfPressure, add_ru, add_PWP)
    )

    # Append original timesteps to added timesteps and shift the time correctly
    first_timesteps = add_Time
    last_timesteps = first_timesteps[-1, 0] + df["Time [sec]"].to_numpy()
    last_timesteps = np.reshape(last_timesteps, (last_timesteps.shape[0], 1))
    full_timesteps = np.vstack((first_timesteps, last_timesteps))
    # Make it df
    header = df.columns
    df_add_data = pd.DataFrame(add_data, columns=header)

    # Append original df to added df
    df_augmented = df_add_data.append(df, ignore_index=True)
    df_augmented["Time [sec]"] = full_timesteps

    return df_augmented


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
                    # read relative density info
                    with open(file) as f:
                        data = f.readlines()[num_headers-1]
                    relative_density = float(data[18:22])
                    # convert csv to dataframe
                    df = pd.read_csv(file, header=num_headers)
                    # add relative density to the first column of the dataframe.
                    df.insert(loc=0, column="Dr [%]", value=relative_density)
                    # compute and insert ru at the 5th column of the dataframes
                    # confining pressure value of the test is located at the first row of effective vertical stress column.
                    confining_pressure = df['Effective Vertical Stress [kPa]'][0]
                    ru = df['Excess Pore Pressure [kPa]']/confining_pressure
                    df.insert(loc=5, column="ru", value=ru)
                    # Augment data by inserting before-CSS test result before the original result
                    df = augment_data(df=df)

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

    savedir = input['paths']['plot']
    os.makedirs(savedir, exist_ok=True)
    plt.savefig(f"{savedir}/exp{exp_id}trial{trial_id}.png")