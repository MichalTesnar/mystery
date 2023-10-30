from src.dataset import Dataset
from src.model import AOIModel
from src.metrics import Metrics

dataset = Dataset()
model = AOIModel()
metrics = Metrics()

while dataset.data_available():
    new_point = dataset.get_new_point()
    flag = model.update_own_training_set(new_point)
    while not flag:
        new_point = dataset.get_new_point()
        flag = model.update_own_training_set(new_point)
    model.retrain()
    metrics.collect_metrics(model)

metrics.plot()
metrics.save()