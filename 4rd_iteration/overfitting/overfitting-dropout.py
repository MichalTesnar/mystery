"""
Checking how the model perform to figure out for how many epochs to train.
"""

import numpy as np
import math

from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import ReduceLROnPlateau
import tensorflow as tf
import keras

from keras_uncertainty.models import StochasticRegressor
from keras_uncertainty.layers import StochasticDropout

import matplotlib.pyplot as plt

NUM_SAMPLES = 10  # number of samples for the network when it runs estimation
# number of epochs to (re)fit the model on the newly observed data
EPOCHS = 100
SAMPLE_RATE = 125  # the rate at which we sample the interval we want to train on
ITERATIONS = 10  # iterations to be plotted
RATE = 0.8  # how much data into the training set


# Approximated function
def toy_function(input):
    output = []
    for inp in input:
        std = max(0.15 / (1.0 + math.exp(-inp)), 0)
        out = math.sin(inp)  # + np.random.normal(0, std)
        output.append(10 * out)
    return np.array(output)

# model definition




def get_dropout_model(prob=0.2):
    model = Sequential()
    model.add(Dense(128, activation="relu", input_shape=(1,)))
    model.add(StochasticDropout(prob))
    model.add(Dense(128, activation="relu"))
    model.add(StochasticDropout(prob))
    model.add(Dense(128, activation="relu"))
    model.add(StochasticDropout(prob))
    model.add(Dense(64, activation="relu"))
    model.add(StochasticDropout(prob))
    model.add(Dense(1, activation="linear"))

    # optimizer = keras.optimizers.Adam(learning_rate=0.0001)
    # model.compile(loss="mean_squared_error", optimizer=optimizer)
    model.compile(loss="mean_squared_error", optimizer='adam')
    # from keras import backend as K
    # K.set_value(model.optimizer.learning_rate, 0.001)
    return model

# Small model
# def get_dropout_model(prob=0.2):
#     model = Sequential()
#     # model.add(Dense(128, activation="relu", input_shape=(1,)))
#     # model.add(StochasticDropout(prob))
#     # model.add(Dense(128, activation="relu"))
#     # model.add(StochasticDropout(prob))
#     # model.add(Dense(128, activation="relu"))
#     # model.add(StochasticDropout(prob))
#     # model.add(Dense(64, activation="relu"))
#     # model.add(StochasticDropout(prob))
#     model.add(Dense(32, activation="relu", input_shape=(1,)))
#     model.add(StochasticDropout(prob))
#     model.add(Dense(32, activation="relu"))
#     model.add(StochasticDropout(prob))
#     model.add(Dense(1, activation="linear"))
#     model.compile(loss="mean_squared_error", optimizer="adam")
#     return model


def get_pred(model, point):
    mc_model = StochasticRegressor(model)
    pred_mean, pred_std = mc_model.predict(point, num_samples=NUM_SAMPLES)
    return pred_mean, pred_std


if __name__ == "__main__":
    history_val = history_train = pred_means = []
    for i in range(ITERATIONS):
        data = np.linspace(-4.0, 4.0, num=SAMPLE_RATE)
        indices = np.arange(SAMPLE_RATE)
        np.random.shuffle(indices)
        train_indices, val_indices = indices[:int(
            SAMPLE_RATE*RATE)], indices[int(SAMPLE_RATE*RATE):]
        # train_indices = np.arange(0, int(SAMPLE_RATE*RATE))*int(SAMPLE_RATE*RATE)
        # val_indices = indices
        x_train = data[train_indices]
        y_train = toy_function(x_train)
        x_val = data[val_indices]
        y_val = toy_function(x_val)
        print("starting iteration: ", i+1, "/", ITERATIONS, sep="")

        model = get_dropout_model()
        # reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.3, patience=15, verbose=1)
        history = model.fit(x_train, y_train, epochs=EPOCHS, batch_size=32, validation_data=(x_val, y_val), verbose=False)#, callbacks=[reduce_lr])
        if len(history_val) != 0:
            history_val += np.array(history.history['val_loss'])
            history_train += np.array(history.history['loss'])
        else:
            history_val = np.array(history.history['val_loss'])
            history_train = np.array(history.history['loss'])

        # fig, ax = plt.subplots(nrows=1, ncols=len(["Dropout"]), figsize=(20, 3))
        # domain = np.linspace(-7.0, 7.0, num=1000)
        # domain = domain.reshape((-1, 1))
        # domain_y = toy_function(domain)
        # points = model.predict(domain)
        # # compute metrics
        # # print(f"score: {score:.2f} calib_err: {calib_err:.2f}")
        # # plot data
        # ax.plot(domain, domain_y, '.', color=(0, 0, 0.9, 1), markersize=10,label= "ground truth")
        # # update plot
        # ax.plot(domain, points, '.', color=(0, 0, 0, 1), markersize=5, label= "prediction")
        # ax.plot(x_train, y_train, '.', color=(0.9, 0, 0, 1), markersize=10, label= "train data")
        # ax.legend(bbox_to_anchor=(1, 1), loc='upper left')
        # plt.savefig(f"performance{i}")

    # Plot training and validation error (loss)
    plt.figure(figsize=(10, 6))
    plt.plot(history_train/ITERATIONS, label='Training Loss')
    plt.plot(history_val/ITERATIONS, label='Validation Loss')
    # plt.plot(pred_means, label='Pred Means')

    x = [np.argmin(history_val)]
    y = [np.min(history_val)/ITERATIONS]
    plt.axvline(x=x, color='b', label='Minimal Validation Loss')
    plt.scatter(x, y, c=x, cmap='hot')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.xticks(range(0, EPOCHS, 25))
    plt.legend()
    plt.title(
        f"Training and Validation Error Curves \n Training on {int(SAMPLE_RATE*RATE)} Points, Validating on {int(SAMPLE_RATE - RATE*SAMPLE_RATE)}")
    plt.show()
