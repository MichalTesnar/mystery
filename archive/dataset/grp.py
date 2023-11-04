from extract import Dataset
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from big_grp import BigGRP

# Dataset
datasetSQ = Dataset(mode="subsampled_sequential", size=0.1)
X_train, y_train = datasetSQ.get_training_set()
X_val, y_val = datasetSQ.get_validation_set()
X_test, y_test = datasetSQ.get_test_set()


# # Model definition
input_dim = X_train.shape[1]
model = BigGRP(input_dim)
# Training


setter_X_train = X_train.iloc[0]
setter_y_train = y_train.iloc[0]


print("Baseline Results")
baseline = BigGRP(input_dim)
baseline.retrain([setter_X_train.values], [setter_y_train.values])
initial_loss = baseline.loss(X_test.values, y_test.values)
baseline.retrain(X_train.values, y_train.values)
final_loss = baseline.loss(X_test.values, y_test.values)

PER_ITER = 20
ITERATIONS = X_train.shape[0] - PER_ITER

def end():
    print("Skipper Results")
    print(skip_counters)
    print(losses)
    print("Intial loss", initial_loss)
    print("Final loss", final_loss)

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
        end()
        exit()

    while np.mean(pred_std) < THRESHOLD:
        data_index += 1
        skip_counter += 1
        current_x = X_train.iloc[data_index]
        current_y = y_train.iloc[data_index]
        pred_mean, pred_std = model.predict([current_x.to_numpy()])
        if data_index == ITERATIONS:
            end()
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

