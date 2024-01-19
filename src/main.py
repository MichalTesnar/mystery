from metrics import Metrics
from model import AIOModel
from dataset import DagonAUVDataset, SinusiodToyExample
import os
import time
import numpy as np
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
tf.get_logger().setLevel('INFO')

np.random.seed(107)

DATASET_TYPE = "Dagon" 
# DATASET_TYPE = "Toy"
EXP_TYPE = "Offline"
EXP_TYPE = "Online"

experiment_specification = {
    "EXPERIMENT_IDENTIFIER": "flipout test",
    "EXPERIMENT_TYPE": DATASET_TYPE,
    "UQ_MODEL": "FLIPOUT",
    "BUFFER_SIZE": 100,
    "MODEL_MODE": "RIRO",
    "DATASET_MODE": "subsampled_sequential",
    "NUMBER_OF_LAYERS": 1,
    "UNITS_PER_LAYER": 16,
    "DATASET_SIZE": 0.05,
    "LEARNING_RATE": 0.01,
    "BATCH_SIZE": 4,
    "PATIENCE": 9,
    "MAX_EPOCHS": 100,
    "ACCEPT_PROBABILITY": 0.5,
    "INPUT_LAYER_SIZE": 6 if DATASET_TYPE == "Dagon" else 1,
    "OUTPUT_LAYER_SIZE": 3 if DATASET_TYPE == "Dagon" else 1,
    "UNCERTAINTY_THRESHOLD": 0.1,
    "NUMBER_OF_ESTIMATORS": 10
}

if experiment_specification["EXPERIMENT_TYPE"] == "Dagon":
    dataset = DagonAUVDataset(experiment_specification)
elif experiment_specification["EXPERIMENT_TYPE"] == "Toy":
    dataset = SinusiodToyExample(experiment_specification)

if experiment_specification["MODEL_MODE"] != "OFFLINE":
    model = AIOModel(dataset.give_initial_training_set(
        experiment_specification["BUFFER_SIZE"]), experiment_specification)
    metrics = Metrics(dataset.get_current_training_set_size,  # account for extra iteration at the end
                      experiment_specification, dataset.get_test_set)
    start_time = time.time()
    training_flag = True
    while dataset.data_available():
        if training_flag:
            history = model.retrain()
            metrics.collect_metrics(model)
            if DATASET_TYPE == "Toy":
                metrics.extra_plots(model)
        training_flag = False
        while not training_flag and dataset.data_available(verbose=True, start_time=start_time):
            new_point = dataset.get_new_point()
            metrics.collect_uncertainty(model, new_point)
            training_flag = model.update_own_training_set(new_point)
            if not training_flag and dataset.data_available():
                metrics.pad_metrics()

    dataset.data_available(verbose=True)
    metrics.save()

else:
    model = AIOModel(dataset.get_training_set, experiment_specification)
    metrics = Metrics(1, experiment_specification, dataset.get_test_set)
    model.retrain(verbose=True)
    metrics.collect_metrics(model)
    print(metrics.metrics_results)
    metrics.save()