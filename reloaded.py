from src.metrics import Metrics

experiment_specification={"EXPERIMENT_IDENTIFIER": "sinus hyperparams THRESHOLD_GREEDY 0.3 (0)"}

metrics = Metrics(experiment_specification=experiment_specification, load=True)

metrics.restore_cummulativeMSE()

metrics.plot()

metrics.save()