import tensorflow as tf
from src.metrics import Metrics
from src.model import AIOModel
from src.dataset import DagonAUVDataset, SinusiodToyExample
import os
import time
import numpy as np
import sys
import random
from src.utils import get_best_params, print_best_params
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.get_logger().setLevel('INFO')

# random.seed(68)

# np.random.seed(42)

# UQ_MODEL = "DROPOUT"
UQ_MODEL = "FLIPOUT"

MODEL_MODE = sys.argv[1]
EXTRA_PARAM = ""

ACCEPT_PROBABILITY = 0.7
UNCERTAINTY_THRESHOLD = 0.02
FLIPOUT_PRIOR_PI = 0.6

if "FLIPOUT" not in UQ_MODEL:
    if MODEL_MODE == "RIRO" and len(sys.argv) > 2:
        ACCEPT_PROBABILITY = float(sys.argv[2])
        EXTRA_PARAM = ACCEPT_PROBABILITY
    if "THRESHOLD" in MODEL_MODE and len(sys.argv) > 2:
        UNCERTAINTY_THRESHOLD = float(sys.argv[2])
        EXTRA_PARAM = UNCERTAINTY_THRESHOLD
elif len(sys.argv) > 2:
    FLIPOUT_PRIOR_PI = float(sys.argv[2])
    EXTRA_PARAM = FLIPOUT_PRIOR_PI


identifier = "Full data fix"
directory = f"hyperparams/{identifier} {MODEL_MODE}"
best_hps = get_best_params(directory)
print_best_params(best_hps)
DATASET_TYPE = "Dagon"  # "Toy"



epochs = 100
if MODEL_MODE == "OFFLINE":
    epochs = 100*7000
elif UQ_MODEL == "FLIPOUT":
    epochs = 300

normalize = False
if UQ_MODEL == "FLIPOUT":
    normalize = True

experiment_specification = {
    "EXPERIMENT_IDENTIFIER": f"{UQ_MODEL} {identifier} {MODEL_MODE} {EXTRA_PARAM} tuned",
    "EXPERIMENT_TYPE": DATASET_TYPE,
    "BUFFER_SIZE": 100,
    "UQ_MODEL": UQ_MODEL,
    "FLIPOUT_PRIOR_PI": FLIPOUT_PRIOR_PI,
    "MODEL_MODE": MODEL_MODE,
    "DATASET_MODE": "subsampled_sequential",
    "NUMBER_OF_LAYERS": best_hps['num_layers'],
    "UNITS_PER_LAYER": best_hps['units'],
    "DATASET_SIZE": 1,
    "LEARNING_RATE": best_hps['learning_rate'],
    "BATCH_SIZE": best_hps['batch_size'],
    "PATIENCE": best_hps['patience'],
    "MAX_EPOCHS": epochs,
    "ACCEPT_PROBABILITY": ACCEPT_PROBABILITY,
    "INPUT_LAYER_SIZE": 6 if DATASET_TYPE == "Dagon" else 1,
    "OUTPUT_LAYER_SIZE": 3 if DATASET_TYPE == "Dagon" else 1,
    "UNCERTAINTY_THRESHOLD": UNCERTAINTY_THRESHOLD,
    "NUMBER_OF_ESTIMATORS": 10
}

if experiment_specification["EXPERIMENT_TYPE"] == "Dagon":
    dataset = DagonAUVDataset(experiment_specification, normalize=normalize)
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
    
        metrics.save()
        
    dataset.data_available(verbose=True)

else:
    model = AIOModel(dataset.get_training_set, experiment_specification)
    metrics = Metrics(1, experiment_specification, dataset.get_test_set)
    model.retrain(verbose=True)
    metrics.collect_metrics(model)
    print(metrics.metrics_results)
    metrics.save()
