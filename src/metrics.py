import pickle
import json
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.metrics import r2_score

"""
Calculation of the metrics of the model.
Saves the values as a pickle, also the graph and the specification of the model.
"""

class Metrics():
    def __init__(self, identifier, experiment_specification={}, test_set=[], load=False) -> None:
        if load:
            self.dir_name = f"results/{identifier}"
            print(f"Loading the results saved in {self.dir_name}")
            self.load()
        
        else:
            dir_i = 0
            while os.path.isdir(f"results/{identifier} ({dir_i})"):
                dir_i += 1
            self.dir_name = f"results/{identifier} ({dir_i})"
            os.mkdir(self.dir_name)
            print(f"The results will be saved in {self.dir_name}")

            self.metrics_results = {"MSE": [],
                                    "R2": []}
        self.model_specification = experiment_specification
        self.identifier = identifier
        self._test_X, self._test_y = test_set
        self.test_set_size = len(self._test_X)

    def collect_metrics(self, model):
        """
        Apply the metrics and save the results.
        """
        pred_mean, pred_std = model.predict(self._test_X)
        for metric in self.metrics_results.keys():
            self.metrics_results[metric].append(self.calculate_metric(metric, pred_mean))
            
    def calculate_metric(self, key, pred_mean):
        if key == "MSE":
            return np.sum(np.square(self._test_y - pred_mean))/self.test_set_size
        if key == "R2":
            return r2_score(pred_mean, self._test_y)

    def plot(self):
        """
        Plot your metrics.
        """
        fig, axs = plt.subplots(len(self.metrics_results.keys()), 1, figsize=(16, 12))
        fig.suptitle('Online Learning Metrics', fontsize=16)
        for i, metric in enumerate(self.metrics_results.keys()):
            y = self.metrics_results[metric]
            x = np.arange(0, len(y))
            axs[i].plot(x, y, label=metric)
            axs[i].set_title(metric)
            axs[i].legend()
        
        plt.tight_layout()
        plt.savefig(f"{self.dir_name}/plotted_metrics")
        # plt.show()
        plt.close()

    def save(self):
        """
        Save the data collected in the dictionary.
        """
        print(f"The results are saved in {self.dir_name}")
        with open(f"{self.dir_name}/metrics_results.pkl", 'wb') as file:
            pickle.dump(self.metrics_results, file)
        with open(f"{self.dir_name}/model_specification.json", "w") as file: 
            json.dump(self.model_specification, file)

    def load(self):
        """
        Reload previously collected metrics from a file.
        """
        with open(f"{self.dir_name}/metrics_results.pkl", 'rb') as file:
            self.metrics_results = pickle.load(file)
        with open(f"{self.dir_name}/model_specification.json", 'r') as file:
            self.model_specification = json.load(file)
        