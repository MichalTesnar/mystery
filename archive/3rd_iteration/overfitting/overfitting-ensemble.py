"""
Checking how the model perform to figure out for how many epochs to train.
"""

import numpy as np
import math

from keras.models import Sequential, Model
from keras.layers import Dense, Input

from keras_uncertainty.models import StochasticRegressor, DeepEnsembleRegressor
from keras_uncertainty.layers import StochasticDropout

from keras_uncertainty.metrics import gaussian_interval_score
from keras_uncertainty.utils import regressor_calibration_error
from keras_uncertainty.losses import regression_gaussian_nll_loss

# fixing ensembles
from tensorflow.python.framework.ops import disable_eager_execution
disable_eager_execution()

import matplotlib.pyplot as plt


NUM_SAMPLES = 10 # number of samples for the network when it runs estimation
EPOCHS = 200 # number of epochs to (re)fit the model on the newly observed data
SAMPLE_RATE = 125 # the rate at which we sample the interval we want to train on
ITERATIONS = 10 # iterations to be plotted
RATE = 0.8 # how much data into the training set


## Approximated function
def toy_function(input):
    output = []
    for inp in input:
        std = max(0.15 / (1.0 + math.exp(-inp)), 0)
        out = math.sin(inp) #+ np.random.normal(0, std)
        output.append(10 * out)
    return np.array(output)

def get_ensemble_model():
    def model_fn():
        inp = Input(shape=(1,))
        x = Dense(128, activation="relu")(inp)
        x = Dense(128, activation="relu")(x)
        x = Dense(64, activation="relu")(x)
        x = Dense(32, activation="relu")(x)
        mean = Dense(1, activation="linear")(x)
        var = Dense(1, activation="softplus")(x)
        ### Small model
        # x = Dense(32, activation="relu")(inp)
        # x = Dense(32, activation="relu")(x)
        # mean = Dense(1, activation="linear")(x)
        # var = Dense(1, activation="softplus")(x)

        train_model = Model(inp, mean)
        pred_model = Model(inp, [mean, var])

        train_model.compile(loss=regression_gaussian_nll_loss(var), optimizer="adam")

        return train_model, pred_model
    
    model = DeepEnsembleRegressor(model_fn, num_estimators=10)
    return model

def pred_ensembles(model, domain):
    pred_mean, pred_std = model.predict(domain)
    return pred_mean, pred_std

def retrain_ensembles(model, x_train, y_train):
    model.fit(x_train, y_train, verbose=True, epochs=EPOCHS)
    return model

if __name__ == "__main__":
    history_val = history_train = pred_means = []
    for i in range(ITERATIONS):
        data = np.linspace(-4.0, 4.0, num=SAMPLE_RATE)
        indices = np.arange(SAMPLE_RATE)
        np.random.shuffle(indices)
        train_indices, val_indices = indices[:int(SAMPLE_RATE*RATE)], indices[int(SAMPLE_RATE*RATE):]
        # train_indices = np.arange(0, int(SAMPLE_RATE*RATE))*int(SAMPLE_RATE*RATE)
        # val_indices = indices
        x_train = data[train_indices]
        y_train = toy_function(x_train)
        x_val = data[val_indices]
        y_val = toy_function(x_val)    
        print("starting iteration: ", i+1,"/", ITERATIONS, sep="")
        

        ######### !!!! CHANGE THE TRAINING OF THE REGRESSOR TO GIVE YOU HISTORY !!!! ######################
        model = get_ensemble_model()
        (history_v, history_t) = model.fit(x_train, y_train, epochs=EPOCHS, validation_data=(x_val, y_val), verbose=False)
        if len(history_val) != 0:
            history_val += np.array(history_v)
            history_train += np.array(history_t)
        else:
            history_val = np.array(history_v)
            history_train = np.array(history_t)
        
        fig, ax = plt.subplots(nrows=1, ncols=len(["Dropout"]), figsize=(20, 3))
        domain = np.linspace(-7.0, 7.0, num=1000)
        domain = domain.reshape((-1, 1))
        domain_y = toy_function(domain)
        pred_mean, pred_std = pred_ensembles(model, domain)

        # points = model.predict(domain)
        # compute metrics        
        # print(f"score: {score:.2f} calib_err: {calib_err:.2f}")
        # plot data
        ax.plot(domain, domain_y, '.', color=(0, 0, 0.9, 1), markersize=10,label= "ground truth")
        # update plot
        ax.plot(x_train, y_train, '.', color=(0.9, 0, 0, 1), markersize=10, label= "train data")
        ax.plot(domain, pred_mean, '.', color=(0, 0, 0, 1), markersize=5, label= "prediction")
        ax.legend(bbox_to_anchor=(1, 1), loc='upper left')
        plt.savefig(f"performance{i}")

    # Plot training and validation error (loss)
    plt.figure(figsize=(10, 6))
    plt.plot(history_train/ITERATIONS, label='Training Loss')
    plt.plot(history_val/ITERATIONS, label='Validation Loss')
    # plt.plot(pred_means, label='Pred Means')

    x = [np.argmin(history_val)]
    y = [np.min(history_val)/ITERATIONS]
    plt.axvline(x=x, color = 'b', label = 'Minimal Validation Loss')
    plt.scatter(x, y, c=x, cmap='hot')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.xticks(range(0, EPOCHS, 25))
    plt.legend()
    plt.title(f"Training and Validation Error Curves \n Training on {int(SAMPLE_RATE*RATE)} Points, Validating on {int(SAMPLE_RATE - RATE*SAMPLE_RATE)}")
    plt.show()

    