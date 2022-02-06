# %%
import json
import matplotlib.pyplot as plt
import model

import preparedata
import result_plot_tools
import numpy as np

# %% import data and dataframe

# open the `input.json` file
import result_plot_tools

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

# %% set variables for train and test

# train
train_x_rnns = rnn_data_train["x_rnns"]  # variables for plotting
train_y_rnns = rnn_data_train["y_rnns"]
train_x_rnn_concat = np.concatenate(train_x_rnns, 0)  # variables for training model
train_y_rnn_concat = np.concatenate(train_y_rnns, 0)

# test
test_x_rnns = rnn_data_test["x_rnns"]  # variables for plotting
test_y_rnns = rnn_data_test["y_rnns"]
test_x_rnn_concat = np.concatenate(test_x_rnns, 0)  # variables for training model
test_y_rnn_concat = np.concatenate(test_y_rnns, 0)

# %% shuffle train set

shuffler = np.random.permutation(len(train_x_rnn_concat))
train_x_rnn_concat_sf = train_x_rnn_concat[shuffler]
train_y_rnn_concat_sf = train_y_rnn_concat[shuffler]

# %% build model

# build a model
lstm_model = model.build_model(window_length=input["window_length"], num_features=len(input["features"]))

# show model summary
lstm_model.summary()

# %% compile and fit
history = model.compile_and_fit(
    input=input, model=lstm_model,
    train_x=test_x_rnn_concat, train_y=test_y_rnn_concat,
)

# %% training history

plt.figure()
plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='val')
plt.legend()
plt.xlabel("Epoch")
plt.ylabel(f"Loss ({input['compile_options']['metric']})")
plt.savefig(f"{input['paths']['plot']}/training_history")

# %% show results
import tensorflow as tf

lstm_model = tf.keras.models.load_model(input['paths']['model'])

train_datasets = result_plot_tools.plot_dataset(
    save_name="train_datasets",
    features=rnn_data_train['feature_datasets'],
    targets=rnn_data_train['target_datasets'],
    ids=train_ids,
    legends=input["features"]+input["targets"],
    subplot_ncols=3
)

test_datasets = result_plot_tools.plot_dataset(
    save_name="test_datasets",
    features=rnn_data_test['feature_datasets'],
    targets=rnn_data_test['target_datasets'],
    ids=test_ids,
    legends=input["features"]+input["targets"],
    subplot_ncols=3
)

# %% make prediction and plot the result

train_predictions = list()
test_predictions = list()

# get prediction with trial datasets
for train_x_rnn in train_x_rnns:
    train_prediction = lstm_model.predict(train_x_rnn)
    train_predictions.append(train_prediction)

# get prediction with test datasets
for test_x_rnn in test_x_rnns:
    test_prediction = lstm_model.predict(test_x_rnn)
    test_predictions.append(test_prediction)

prediction_train = result_plot_tools.plot_prediction(
    save_name="prediction_train",
    targets=train_y_rnns,
    prediction=train_predictions,
    ids=train_ids,
    subplot_ncols=3)

#%%
plt.plot(train_y_rnns[1])
plt.plot(train_predictions[1])


