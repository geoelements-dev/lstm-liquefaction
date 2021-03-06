from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras import optimizers
import os
import json

input = json.load(open("input.json", "r"))


def build_model(window_length, num_features):
    # setup the model
    model = Sequential()
    model.add(
        layers.LSTM(units=128, return_sequences=True, input_shape=(window_length, num_features), activation='tanh'))
    # model.add(Dropout(0.1))
    model.add(layers.LSTM(units=128, activation='tanh'))
    # model.add(Dropout(0.1))
    model.add(layers.Dense(64, activation='tanh'))
    # model.add(Dropout(0.1))
    model.add(layers.Dense(16, activation='tanh'))
    # model.add(Dropout(0.1))
    model.add(layers.Dense(1, activation='tanh'))

    return model


def compile_and_fit(input, model, train_x, train_y):
    # callback setting
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=1000, mode='min'),
        ModelCheckpoint(filepath=input["paths"]["check_point"], mode='auto')
    ]

    # compile
    model.compile(loss=input["compile_options"]["loss"],
                  optimizer=optimizers.Adam(learning_rate=input["compile_options"]["learning_rate"]),
                  metrics=input["compile_options"]["metric"])

    # fit the model
    history = model.fit(train_x, train_y,
                        epochs=input["fit_options"]["epochs"], batch_size=input["fit_options"]["batch_size"], verbose=2,
                        validation_split=0.20,
                        callbacks=callbacks
                        )

    savedir = os.path.join(input["paths"]["model"])
    # os.makedirs(savedir, exist_ok=True)
    model.save(savedir)

    return history
