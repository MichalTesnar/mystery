from src.metrics import Metrics

experiment_specification={"EXPERIMENT_IDENTIFIER": "firo test (0)"}

metrics = Metrics(experiment_specification=experiment_specification, load=True)

# metrics.restore_cummulativeMSE()

metrics.plot()

metrics.save()