"""
Adaptation of the model to learn from the 10 last seen samples.
We introduce 1 new sample at the time.
"""

import numpy as np
import math

from keras.models import Sequential
from keras.layers import Dense

from keras_uncertainty.models import StochasticRegressor
from keras_uncertainty.layers import StochasticDropout

from keras_uncertainty.metrics import gaussian_interval_score
from keras_uncertainty.utils import regressor_calibration_error

import matplotlib.pyplot as plt

## Constants
NUM_SAMPLES = 100 # number of samples for the network when it runs estimation
EPOCHS = 400 # number of epochs to (re)fit the model on the newly observed data
SAMPLE_RATE = 20 # the rate at which we sample the interval we want to train on
NEW_DATA_RATE = 10 # how much data the model sees at the same time

## Approximated function
def toy_function(input):
    output = []
    for inp in input:
        std = max(0.15 / (1.0 + math.exp(-inp)), 0)
        out = math.sin(inp) #+ np.random.normal(0, std)
        output.append(10 * out)
    return np.array(output)


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
    model.compile(loss="mean_squared_error", optimizer="adam")
    return model

## trains the model, or retrains the model on the new data, then samples is to obtain the prediction
def retrain_dropout_model(model, x_train, y_train, domain):
    # training the model
    model.fit(x_train, y_train, verbose=True, epochs=EPOCHS)
    # sampling the model
    mc_model = StochasticRegressor(model)
    pred_mean, pred_std = mc_model.predict(domain, num_samples=NUM_SAMPLES)
    return pred_mean, pred_std, model

if __name__ == "__main__":
    # data
    x_train = np.linspace(-4.0, 4.0, num=2*SAMPLE_RATE)
    # train_indices = np.arange()
    # train_indices = np.random.choice(2*SAMPLE_RATE, SAMPLE_RATE, replace=False)
    train_indices = np.append(np.arange(0, 9), np.arange(39, 40))
    # train_indices = np.append(np.arange(0, 50), np.arange(150,199))
    np.random.shuffle(train_indices)
    x_train = x_train[train_indices]
    y_train = toy_function(x_train)
    domain = np.linspace(-7.0, 7.0, num=100)
    domain = domain.reshape((-1, 1))
    domain_y = toy_function(domain)
    # plotting
    fig, ax = plt.subplots(nrows=1, ncols=len(["Dropout"]), figsize=(20, 3))
    ax.set_title(f"MC DropOut 9 + 1 Test, {EPOCHS} epochs")
    ax.set_ylim([-20.0, 20.0])
    ax.axvline(x=-4.0, color="black", linestyle="dashed")
    ax.axvline(x= 4.0, color="black", linestyle="dashed")
    ax.get_xaxis().set_ticks([])
    ax.get_yaxis().set_ticks([])    
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    # define model to be retrained later
    current_model = get_dropout_model()
    
    # iteratively retrain on new data
    current_x_train, current_y_train = x_train, y_train
    y_pred_mean, y_pred_std, current_model = retrain_dropout_model(current_model, current_x_train, current_y_train, domain)
    points = current_model.predict(domain)
    # compute metrics        
    score = gaussian_interval_score(domain_y, y_pred_mean, y_pred_std)
    calib_err = regressor_calibration_error(y_pred_mean, domain_y, y_pred_std)
    # print(f"score: {score:.2f} calib_err: {calib_err:.2f}")
    # plot data
    y_pred_mean = y_pred_mean.reshape((-1,))
    y_pred_std = y_pred_std.reshape((-1,))
    y_pred_up_1 = y_pred_mean + y_pred_std
    y_pred_down_1 = y_pred_mean - y_pred_std
    ax.plot(current_x_train, current_y_train, '.', color=(0.9, 0, 0, 1), markersize=5, label="Trainng Set")
    ax.plot(domain, domain_y, '.', color=(0, 0.9, 0, 1), markersize=3, label="Target Function")
    # ax.plot(domain, points, '.', color=(0.9, 0.9, 0.9, 1), markersize=5)
    ax.fill_between(domain.ravel(), y_pred_down_1, y_pred_up_1,  color=(0, 0.5, 0.9, 1))
    ax.plot(domain.ravel(), y_pred_mean, '.', color=(1, 1, 1, 0.8), markersize=0.2)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    # update plot
    plt.show()