from src.metrics import Metrics
from src.model import AIOModel
from src.dataset import DagonAUVDataset, SinusiodToyExample
import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


experiment_specification = {
    "EXPERIMENT_IDENTIFIER": "riro 0.3 test",
    "BUFFER_SIZE": 100,
    "MODEL_MODE": "RIRO",
    "DATASET_MODE": "subsampled_sequential",
    "DATASET_SIZE": 0.1,
    "BATCH_SIZE": 2,
    "PATIENCE": 10,
    "MAX_EPOCHS": 1000,
    "ACCEPT_PROBABILITY": 0.3,
    "INPUT_LAYER_SIZE": 1,
    "OUTPUT_LAYER_SIZE": 1,
    "UNCERTAINTY_THRESHOLD": 0.1
}

# dataset = DagonAUVDataset(experiment_specification)
dataset = SinusiodToyExample(experiment_specification)
model = AIOModel(dataset.give_initial_training_set(
    experiment_specification["BUFFER_SIZE"]), experiment_specification)
metrics = Metrics(dataset.get_current_training_set_size,
                  experiment_specification, dataset.get_test_set)

training_flag = True
while dataset.data_available():
    if training_flag:
        model.retrain()
        metrics.collect_metrics(model)
        metrics.extra_plots(model)
    training_flag = False
    while not training_flag and dataset.data_available(verbose=True):
        new_point = dataset.get_new_point()
        training_flag = model.update_own_training_set(new_point)
        if not training_flag:
            metrics.pad_metrics()

metrics.plot()
metrics.save()
