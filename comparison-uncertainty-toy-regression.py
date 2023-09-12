import numpy as np
import math

import keras
from keras.models import Sequential, Model
from keras.layers import Dense, Input

import keras_uncertainty
from keras_uncertainty.models import DeepEnsembleRegressor, StochasticRegressor, TwoHeadStochasticRegressor
from keras_uncertainty.layers import DropConnectDense, VariationalDense, FlipoutDense, StochasticDropout
from keras_uncertainty.metrics import gaussian_interval_score
from keras_uncertainty.utils import regressor_calibration_error
from keras_uncertainty.losses import regression_gaussian_nll_loss, regression_gaussian_beta_nll_loss

from sklearn.datasets import make_moons

import matplotlib.pyplot as plt

def toy_function(input):
    output = []

    for inp in input:
        std = max(0.15 / (1.0 + math.exp(-inp)), 0)

        out = math.sin(inp) + np.random.normal(0, std)
        output.append(10 * out)

    return np.array(output)

def train_dropout_model(model, x_train, y_train, domain, prob=0.2, return_model=False):
    if model == None:
        model = Sequential()
        model.add(Dense(32, activation="relu", input_shape=(1,)))
        model.add(StochasticDropout(prob))
        model.add(Dense(32, activation="relu"))
        model.add(StochasticDropout(prob))
        model.add(Dense(1, activation="linear"))
        model.compile(loss="mean_squared_error", optimizer="adam")
    model.fit(x_train, y_train, verbose=False, epochs=1000)
    print("Traning done")
    

    mc_model = StochasticRegressor(model)
    pred_mean, pred_std = mc_model.predict(domain, num_samples=100)
    
    return pred_mean, pred_std, model


NUM_SAMPLES = 30

if __name__ == "__main__":
    SAMPLE_RATE = 1000
    x_train = np.linspace(-4.0, 4.0, num=SAMPLE_RATE)
    # shuffle the data
    train_indices = np.arange(SAMPLE_RATE)
    np.random.shuffle(train_indices)
    x_train = x_train[train_indices]
    y_train = toy_function(x_train)
    # domain stuff
    domain = np.linspace(-7.0, 7.0, num=100)
    domain = domain.reshape((-1, 1))
    domain_y = toy_function(domain)

    plt.ion()
    fig, ax = plt.subplots(nrows=1, ncols=len(["Dropout"]), figsize=(20, 3))
    NEW_DATA_RATE = 3
    current_model = None

    for i in range(int(SAMPLE_RATE/NEW_DATA_RATE)-1):
        start, end = (i)*NEW_DATA_RATE, (i+1)*NEW_DATA_RATE
        current_x_train = x_train[start:end]
        current_y_train = y_train[start:end]
        y_pred_mean, y_pred_std, current_model = train_dropout_model(current_model, current_x_train, current_y_train, domain)
        score = gaussian_interval_score(domain_y, y_pred_mean, y_pred_std)
        calib_err = regressor_calibration_error(y_pred_mean, domain_y, y_pred_std)

        y_pred_mean = y_pred_mean.reshape((-1,))
        y_pred_std = y_pred_std.reshape((-1,))
        y_pred_up_1 = y_pred_mean + y_pred_std
        y_pred_down_1 = y_pred_mean - y_pred_std
        
        ax.plot(current_x_train, current_y_train, '.', color=(0.9, 0.0, 0.0, 1), markersize=15, label="Train Set")

        ax.fill_between(domain.ravel(), y_pred_down_1, y_pred_up_1,  color=(i*0.1, 0, 0.9, 0.5), label="One $\sigma$ CI")
        # ax.plot(domain.ravel(), y_pred_mean, '.', color=(0, 0.9, 0.0, 0.8), markersize=0.2, label="Mean")

        ax.set_title("{}\nIS: {:.2f} CE: {:.2f}".format(1, score, calib_err))
        ax.set_ylim([-20.0, 20.0])

        ax.axvline(x=-4.0, color="black", linestyle="dashed")
        ax.axvline(x= 4.0, color="black", linestyle="dashed")
        ax.get_xaxis().set_ticks([])
        ax.get_yaxis().set_ticks([])    

        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        #plt.savefig("uncertainty-toy-regression.png", bbox_inches="tight")
        #plt.savefig("uncertainty-toy-regression.pdf", bbox_inches="tight")
        fig.canvas.draw()
        fig.canvas.flush_events()