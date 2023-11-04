from src.dataset import Dataset
from src.model import AIOModel
from src.metrics import Metrics

experiment_specification={
    "BUFFER_SIZE": 10,
    "MODEL_MODE": "RIRO",
    "DATASET_MODE": "subsampled_sequential",
    "DATASET_SIZE": 0.01,
    "BATCH_SIZE": 2,
    "PATIENCE": 10,
    "EPOCHS": 100,
    "ACCEPT_PROBABILITY": 0.5
}

"""
@TODO
1. add padding to methods that finished earlier to make the graphs comparable
"""

dataset = Dataset(mode=experiment_specification["DATASET_MODE"], size=experiment_specification["DATASET_SIZE"])
model = AIOModel(dataset.give_initial_training_set(experiment_specification["BUFFER_SIZE"]), experiment_specification)
metrics = Metrics("testing", experiment_specification, dataset.get_test_set)

while dataset.data_available():
    flag = False
    while not flag and dataset.data_available(verbose=True):
        new_point = dataset.get_new_point()
        flag = model.update_own_training_set(new_point)
    if flag:
        model.retrain()
        metrics.collect_metrics(model)

metrics.plot()
metrics.save()