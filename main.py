from src.dataset import Dataset
from src.model import AIOModel
from src.metrics import Metrics

BUFFER_SIZE = 100
MODEL_MODE = "FIFO"
DATASET_MODE = "subsampled_sequential"

dataset = Dataset(mode=DATASET_MODE, size=0.2)
model = AIOModel(dataset.give_initial_training_set(BUFFER_SIZE), mode=MODEL_MODE)
metrics = Metrics(dataset.get_test_set)

while dataset.data_available():
    flag = False
    while not flag and dataset.data_available():
        new_point = dataset.get_new_point()
        flag = model.update_own_training_set(new_point)
    if flag:
        model.retrain()
        metrics.collect_metrics(model)

metrics.plot()
metrics.save()