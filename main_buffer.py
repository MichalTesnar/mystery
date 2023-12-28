import tensorflow as tf
from src.metrics import Metrics
from src.model import AIOModel
from src.dataset import DagonAUVDataset, SinusiodToyExample
import os
import time
import numpy as np
import sys
from src.utils import get_best_params, print_best_params
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.get_logger().setLevel('INFO')

np.random.seed(107)

assert False == True, "HARDCODE BEST VALUES P and T FOR ALL METHODS"

MODEL_MODE = sys.argv[1]
BUFFER_SIZE = sys.argv[2]
EXTRA_PARAM = BUFFER_SIZE

identifier = "Full data"
directory = f"hyperparams/{identifier} {MODEL_MODE}"
best_hps = get_best_params(directory)
print_best_params(best_hps)
DATASET_TYPE = "Dagon"  # "Toy"

experiment_specification = {
    "EXPERIMENT_IDENTIFIER": f"{identifier} {MODEL_MODE} {EXTRA_PARAM} tuned",
    "EXPERIMENT_TYPE": DATASET_TYPE,
    "BUFFER_SIZE": BUFFER_SIZE,
    "MODEL_MODE": MODEL_MODE,
    "DATASET_MODE": "subsampled_sequential",
    "NUMBER_OF_LAYERS": best_hps['num_layers'],
    "UNITS_PER_LAYER": best_hps['units'],
    "DATASET_SIZE": 1,
    "LEARNING_RATE": best_hps['learning_rate'],
    "BATCH_SIZE": best_hps['batch_size'],
    "PATIENCE": best_hps['patience'],
    "MAX_EPOCHS": 100 if MODEL_MODE != "OFFLINE" else 100*7000,
    "ACCEPT_PROBABILITY": ACCEPT_PROBABILITY,
    "INPUT_LAYER_SIZE": 6 if DATASET_TYPE == "Dagon" else 1,
    "OUTPUT_LAYER_SIZE": 3 if DATASET_TYPE == "Dagon" else 1,
    "UNCERTAINTY_THRESHOLD": UNCERTAINTY_THRESHOLD,
    "NUMBER_OF_ESTIMATORS": 10
}

if experiment_specification["EXPERIMENT_TYPE"] == "Dagon":
    dataset = DagonAUVDataset(experiment_specification)
elif experiment_specification["EXPERIMENT_TYPE"] == "Toy":
    dataset = SinusiodToyExample(experiment_specification)

if MODEL_MODE != "OFFLINE":
    model = AIOModel(dataset.give_initial_training_set(
        experiment_specification["BUFFER_SIZE"]), experiment_specification)
    metrics = Metrics(dataset.get_current_training_set_size,  # account for extra iteration at the end
                      experiment_specification, dataset.get_test_set)
    start_time = time.time()
    training_flag = True
    while dataset.data_available():
        if training_flag:
            history = model.retrain()
            # print(history[-1])
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

    # metrics.plot()
    metrics.save()

else:
    model = AIOModel(dataset.get_training_set, experiment_specification)
    metrics = Metrics(1, experiment_specification, dataset.get_test_set)
    # model.retrain(verbose=True)
    metrics.collect_metrics(model)
    print(metrics.metrics_results)
    # metrics.plot()
    metrics.save()
