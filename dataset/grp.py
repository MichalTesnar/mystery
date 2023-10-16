from extract import Dataset
import numpy as np
import matplotlib.pyplot as plt
from big_grp import BigGRP

# Dataset
datasetSQ = Dataset(mode="subsumpled_sequential", size=0.02)
X_train, y_train = datasetSQ.get_training_set()
X_val, y_val = datasetSQ.get_validation_set()
X_test, y_test = datasetSQ.get_test_set()


# # Model definition
input_dim = X_train.shape[1]
model = BigGRP(input_dim)
# Training

PER_ITER = 10
ITERATIONS = 10

########### IMPLEMENT ITERATIONS

for i in range(ITERATIONS):
    model.retrain(X_train, y_train)
    break

pred_mean, pred_std = model.predict([X_test.iloc[0]])
print(pred_mean)
print(y_test.iloc[0])
