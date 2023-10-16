from extract import Dataset
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from big_grp import BigGRP

# Dataset
datasetSQ = Dataset(mode="subsampled_sequential", size=1)
X_train, y_train = datasetSQ.get_training_set()
X_val, y_val = datasetSQ.get_validation_set()
X_test, y_test = datasetSQ.get_test_set()


# # Model definition
input_dim = X_train.shape[1]
model = BigGRP(input_dim)
# Training


PER_ITER = 30
ITERATIONS = X_train.shape[0] - PER_ITER

########### NORMAL ITERATIONS ############
# for iter in range(ITERATIONS):
#     current_X_train, current_y_train = X_train[iter: iter + PER_ITER + 1], y_train[iter: iter + PER_ITER + 1]
#     model.retrain(current_X_train, current_y_train)
#     print(model.loss(X_test, y_test))

# ########### REJECT LOW VARIANCE ############
data_index = 0
THRESHOLD = 0.2
current_X_train, current_y_train = X_train.iloc[: PER_ITER], y_train.iloc[:PER_ITER]
model.retrain(current_X_train, current_y_train)
skip_counters = []
losses = []
iterations = 0
while data_index < ITERATIONS:
    iterations += 1
    current_x = X_train.iloc[data_index]
    current_y = y_train.iloc[data_index]
    pred_mean, pred_std = model.predict([current_x.to_numpy()])
    pred_means, pred_stds = model.predict(X_train)
    idx_to_replace = np.argmin(pred_stds.mean(axis=1))
    skip_counter = 0

    if data_index == ITERATIONS:
        print("Big Oopsie")
        print(skip_counters)
        print(losses)
        exit()

    while np.mean(pred_std) < THRESHOLD:
        data_index += 1
        skip_counter += 1
        current_x = X_train.iloc[data_index]
        current_y = y_train.iloc[data_index]
        pred_mean, pred_std = model.predict([current_x.to_numpy()])
        if data_index == ITERATIONS:
            print("Big Oopsie")
            print(skip_counters)
            print(losses)
            exit()
    skip_counters.append(skip_counter)

    # current_X_train = current_X_train.drop([1])
    # current_X_train = pd.concat([current_X_train, pd.DataFrame([current_x])], ignore_index=True)
    # current_y_train = current_y_train.drop([1])
    # current_y_train = pd.concat([current_y_train, pd.DataFrame([current_y])], ignore_index=True)
    pd.options.mode.chained_assignment = None
    current_X_train.loc[idx_to_replace] = current_x
    current_y_train.loc[idx_to_replace] = current_y

    model.retrain(current_X_train, current_y_train)
    loss = model.loss(X_test, y_test)
    losses.append(loss)
    
