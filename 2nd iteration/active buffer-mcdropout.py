"""
We use active buffer to discard the most certain sample when introducing the newly gathered point.
We see
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
SAMPLE_RATE = 1000 # the rate at which we sample the interval we want to train on
NEW_DATA_RATE = 20 # size of the buffer for both methods
ITERATIONS = 300 # iterations to be plotted
NEW_PER_ITER = 3 # how much data we add each time
PLOT = True

## Approximated function
def toy_function(input):
    output = []
    for inp in input:
        std = max(0.15 / (1.0 + math.exp(-inp)), 0)
        out = math.sin(inp) + np.random.normal(0, std)
        output.append(10 * out)
    return np.array(output)

## model definition
## model definition
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
    model.fit(x_train, y_train, verbose=False, epochs=EPOCHS)
    # sampling the model
    mc_model = StochasticRegressor(model)
    pred_mean, pred_std = mc_model.predict(domain, num_samples=NUM_SAMPLES)
    return pred_mean, pred_std, model

def get_pred(model, point):
    mc_model = StochasticRegressor(model)
    pred_mean, pred_std = mc_model.predict(point, num_samples=NUM_SAMPLES)
    return pred_std

if __name__ == "__main__":
    # data
    x_train = np.linspace(-4.0, 4.0, num=SAMPLE_RATE)
    train_indices = np.arange(SAMPLE_RATE)
    np.random.shuffle(train_indices)
    x_train = x_train[train_indices]
    y_train = toy_function(x_train)
    domain = np.linspace(-4.0, 4.0, num=100)
    domain = domain.reshape((-1, 1))
    domain_y = toy_function(domain)
    # plotting
    # define model to be retrained later
    current_model = get_dropout_model()
    current_x_train, current_y_train = x_train[0:NEW_DATA_RATE], y_train[0:NEW_DATA_RATE] # grab empty at the start
    
    # iteratively retrain on new data
    for i in range(ITERATIONS):
        # obtain input and train the model
        current_x, current_y = x_train[NEW_DATA_RATE + NEW_PER_ITER*i: NEW_DATA_RATE + NEW_PER_ITER*(i+1)], y_train[NEW_DATA_RATE + NEW_PER_ITER*i:NEW_DATA_RATE + NEW_PER_ITER*(i+1)]
        # buffer update step
        predicted_stds = get_pred(current_model, current_x_train.reshape((-1, 1)))
        predicted_std = get_pred(current_model, current_x.reshape((-1, 1)))
        for j, pre in enumerate(predicted_std):
            if min(predicted_stds) < pre:
                idx = np.argmin(predicted_stds)
                predicted_stds[idx] = pre
                current_x_train[idx] = current_x[j]
                current_y_train[idx] = current_y[j]
            # give the model extra push -- extra training on the new point -- OPTIONAL
            # push_x = np.tile(current_x, min(NEW_DATA_RATE, max(i, 1)))
            # push_y = np.tile(current_y, min(NEW_DATA_RATE, max(i, 1)))
            # _, _, current_model = retrain_dropout_model(current_model, push_x, push_y, domain)
        # retrain
        y_pred_mean, y_pred_std, current_model = retrain_dropout_model(current_model, current_x_train, current_y_train, domain)
        # metrics
        score = gaussian_interval_score(domain_y, y_pred_mean, y_pred_std)
        calib_err = regressor_calibration_error(y_pred_mean, domain_y, y_pred_std)
        print(f"iteration: {i} score: {score:.2f} calib_err: {calib_err:.2f}")
        # PLOTTING
        y_pred_mean = y_pred_mean.reshape((-1,))
        y_pred_std = y_pred_std.reshape((-1,))
        y_pred_up_1 = y_pred_mean + y_pred_std
        y_pred_down_1 = y_pred_mean - y_pred_std
        # plot data
        if PLOT:
            fig, ax = plt.subplots(nrows=1, ncols=len(["Dropout"]), figsize=(20, 3))
            ax.set_title("Iterative Active Buffer")
            ax.set_ylim([-20.0, 20.0])
            ax.axvline(x=-4.0, color="black", linestyle="dashed")
            ax.axvline(x= 4.0, color="black", linestyle="dashed")
            ax.get_xaxis().set_ticks([])
            ax.get_yaxis().set_ticks([])    
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            ax.plot(current_x_train, current_y_train, '.', color=(0.9, 0, 0, 0.5), markersize=15, label="training set")
            ax.plot(domain, domain_y, '.', color=(0, 0.9, 0, 1), markersize=3, label="ground truth")
            ax.fill_between(domain.ravel(), y_pred_down_1, y_pred_up_1,  color=(0, 0.5, 0.9, 0.5))
            ax.plot(domain.ravel(), y_pred_mean, '.', color=(1, 1, 1, 0.8), markersize=0.2)
            # update plot
            # fig.canvas.draw()
            # fig.canvas.flush_events()
            plt.savefig(f"pic_{i}i")
            # plt.show()
    if PLOT:
        plt.ioff()
        plt.show()