from src.dataset import Dataset
from src.model import AIOModel
from src.metrics import Metrics

dataset = Dataset()
model = AIOModel()
metrics = Metrics()

while dataset.data_available():
    flag = False
    while not flag:
        new_point = dataset.get_new_point()
        flag = model.update_own_training_set(new_point)
    model.retrain()
    metrics.collect_metrics(model)

metrics.plot()
metrics.save()