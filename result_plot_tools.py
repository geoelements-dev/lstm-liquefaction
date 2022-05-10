from matplotlib import pyplot as plt
import numpy as np
import json

input = json.load(open("input.json", "r"))


def plot_dataset(save_name, features, targets, ids, legends, subplot_ncols=3):

    num_trials = len(features)
    nr = int(np.ceil(num_trials/subplot_ncols))

    fig, axs = plt.subplots(nrows=nr, ncols=subplot_ncols, figsize=(subplot_ncols * 5, nr * 3))
    unrolled_axs = axs.flatten()
    for i in range(num_trials):
        unrolled_axs[i].plot(features[i])
        unrolled_axs[i].plot(targets[i])
        unrolled_axs[i].legend(legends)
        unrolled_axs[i].set_xlabel("Data points")
        unrolled_axs[i].set_ylabel("Normalized values")
        unrolled_axs[i].set_title(f"experiment{ids[i]['exp_id']}-trial{ids[i]['trial_id']}")
    fig.tight_layout()
    global input
    plt.savefig(f"{input['paths']['plot']}/{save_name}.png")


def plot_prediction(save_name, features, targets, prediction, ids, subplot_ncols=3):

    global input

    num_trials = len(targets)  # total num of trials in train set
    nr = int(np.ceil(num_trials / subplot_ncols))  # num of rows in subplot

    fig, axs = plt.subplots(nrows=nr, ncols=subplot_ncols, figsize=(subplot_ncols * 5, nr * 3))
    axs_unroll = axs.flatten()
    for i in range(num_trials):
        axs_unroll[i].plot(features[i][:][input["window_length"]:, 2], c='r', label='input stress')
        axs_unroll[i].plot(targets[i], c='g', label='target')
        axs_unroll[i].plot(prediction[i], c='k', label='prediction')
        axs_unroll[i].legend(loc='best')
        axs_unroll[i].set_xlabel("Data points")
        axs_unroll[i].set_ylabel("Normalized values")
        axs_unroll[i].set_title(f"experiment-{ids[i]['exp_id']}-trial-{ids[i]['trial_id']}")
    fig.tight_layout()

    plt.savefig(f"{input['paths']['plot']}/{save_name}.png")