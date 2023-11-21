from src.metrics import Metrics
from src.model import AIOModel
from src.dataset import DagonAUVDataset, SinusiodToyExample
import os
import time
import numpy as np
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
tf.get_logger().setLevel('INFO')

np.random.seed(107)

"""
SINUS
number of layers 2.
units per layer 53
learning rate 0.1. --> 0.001
batch size 2.
patience 4.

DAGON
number of layers 2.
units per layer 15
learning rate 0.1 --> 0.001
batch size 2.
patience 3.
"""

experiment_specification = {
    "EXPERIMENT_IDENTIFIER": "testing new func",
    "BUFFER_SIZE": 100,
    "MODEL_MODE": "THRESHOLD",
    "DATASET_MODE": "subsampled_sequential",
    "NUMBER_OF_LAYERS": 4,
    "UNITS_PER_LAYER": 32,
    "DATASET_SIZE": 0.1,
    "LEARNING_RATE": 0.01,
    "BATCH_SIZE": 2,
    "PATIENCE": 100,
    "MAX_EPOCHS": 1000,
    "ACCEPT_PROBABILITY": 0.7,
    "INPUT_LAYER_SIZE": 1,
    "OUTPUT_LAYER_SIZE": 1,
    "UNCERTAINTY_THRESHOLD": 0.01,
    "RUNNING_MEAN_WINDOW": 10
}

# dataset = DagonAUVDataset(experiment_specification)
dataset = SinusiodToyExample(experiment_specification)
model = AIOModel(dataset.give_initial_training_set(
    experiment_specification["BUFFER_SIZE"]), experiment_specification)
metrics = Metrics(dataset.get_current_training_set_size, # account for extra iteration at the end
                  experiment_specification, dataset.get_test_set)
start_time = time.time()
training_flag = True
while dataset.data_available():
    if training_flag:
        model.retrain()
        metrics.collect_metrics(model)
        metrics.extra_plots(model)
    training_flag = False
    while not training_flag and dataset.data_available(verbose=True, start_time=start_time):
        new_point = dataset.get_new_point()
        training_flag = model.update_own_training_set(new_point)
        if not training_flag and dataset.data_available():
            metrics.pad_metrics()

dataset.data_available(verbose=True)

metrics.plot()
metrics.save()
