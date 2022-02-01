# %%
import json
import preparedata


# %% import data and dataframe

# open the `input.json` file

input = json.load(open("input.json", "r"))

# %% select exp-trials to use for train & test

train_ids = input["train_ids"]
test_ids = input["test_ids"]

train_dfs = preparedata.select_datasets(dataset_ids=train_ids)
test_dfs = preparedata.select_datasets(dataset_ids=test_ids)

# %% normalize the shear stress with the confining pressure and the time step with the final time for each selected dataset

normalized_train_dfs = preparedata.normalize(dfs=train_dfs)
normalized_test_dfs = preparedata.normalize(dfs=test_dfs)

# %% set features and target, sample the datasets based on time window sampling

rnn_data_train = preparedata.rnn_inputs(
    dfs=normalized_train_dfs, features=input["features"], targets=input["targets"], window_length=input["window_length"]
)
rnn_data_test = preparedata.rnn_inputs(
    dfs=normalized_test_dfs, features=input["features"], targets=input["targets"], window_length=input["window_length"]
)
