from src.dataset import DagonAUVDataset, SinusiodToyExample
from src.model import AIOModel
from src.metrics import Metrics

experiment_specification={
    "EXPERIMENT_IDENTIFIER": "testing",
    "BUFFER_SIZE": 20,
    "MODEL_MODE": "FIFO",
    "DATASET_MODE": "subsampled_sequential",
    "DATASET_SIZE": 0.01,
    "BATCH_SIZE": 2,
    "PATIENCE": 10,
    "MAX_EPOCHS": 1000,
    "ACCEPT_PROBABILITY": 0.5,
    "INPUT_LAYER_SIZE": 1,
    "OUTPUT_LAYER_SIZE": 1
}

"""
@BUG
- learn to use debuggers
- retracing
- debug simple dataset, inspect if it works correctly
@TODO
1. active buffer method
    1.1 greedy
    1.2 threshold
    1.3 greedy threshold
2. dynamic regres (AUC metric)
"""

# dataset = DagonAUVDataset(experiment_specification)
dataset = SinusiodToyExample(experiment_specification)
model = AIOModel(dataset.give_initial_training_set(experiment_specification["BUFFER_SIZE"]), experiment_specification)
metrics = Metrics(dataset.get_training_set_size, experiment_specification, dataset.get_test_set)

while dataset.data_available():
    flag = False
    while not flag and dataset.data_available(verbose=True):
        new_point = dataset.get_new_point()
        flag = model.update_own_training_set(new_point)
        if not flag:
            metrics.pad_metrics()
    if flag:
        model.retrain()
        metrics.collect_metrics(model)

metrics.plot()
metrics.save()