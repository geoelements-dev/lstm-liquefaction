import json
import pickle
import numpy as np
import preparedata

# open the `input.json` file
input = json.load(open("input.json", "r"))

# select dfs at exp-trials to use for train & test
train_ids = input["train_ids"]
test_ids = input["test_ids"]
train_dfs = preparedata.select_datasets(dataset_ids=train_ids)
test_dfs = preparedata.select_datasets(dataset_ids=test_ids)

# normalize the shear stress with the confining pressure and the time step with the final time for each selected dataset
normalized_train_dfs = preparedata.normalize(dfs=train_dfs, select_columns=input["features"])
normalized_test_dfs = preparedata.normalize(dfs=test_dfs, select_columns=input["targets"])

# make `encoder_input_data`
max_train_seq_length = max([len(normalized_train_df) for normalized_train_df in normalized_train_dfs])
max_test_seq_length = max([len(normalized_test_df) for normalized_test_df in normalized_test_dfs])
encoder_input_data = np.zeros(
    (len(normalized_train_dfs), max_train_seq_length, len(input["features"]))
)

# save as `.pkl`
with open('outputs/encoder_input_data.pkl', 'wb') as f:
    pickle.dump(encoder_input_data, f)
