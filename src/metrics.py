import pickle
import numpy as np
import matplotlib.pyplot as plt
import os

"""
@TODO: based on the implementation of .predict implement calculation of metrics
"""

class Metrics():
    def __init__(self, identifier, test_set=[], load=False) -> None:
        if load:
            self.dir_name = f"results/{identifier}"
            print(f"Loading the results saved in {self.dir_name}")
        
        else:
            dir_i = 0
            while os.path.isdir(f"results/{identifier} ({dir_i})"):
                dir_i += 1
            self.dir_name = f"results/{identifier} ({dir_i})"
            os.mkdir(self.dir_name)
            print(f"The results will be saved in {self.dir_name}")

            self.identifier = identifier
            self._test_X, self._test_y = test_set
            self.metrics_results = {"MSE": [],
                                    "R2": []}

    def collect_metrics(self, model):
        """
        Apply the metrics and save the results.
        """
        output = model.predict(self._test_X)
        for metric in self.metrics_results.keys():
            self.metrics_results[metric].append(self.calculate_metric(metric, output))
            
    def calculate_metric(self, key, output):
        if key == "MSE":
            return 0
        if key == "R2":
            return 1

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
        plt.show()
        plt.close()

    def save(self):
        """
        Save the data collected in the dictionary.
        """
        with open(f"{self.dir_name}/metrics_results.pkl", 'wb') as file:
            pickle.dump(self.metrics_results, file)

    def load(self):
        """
        Reload previously collected metrics from a file.
        """
        with open(f"results/{self.dir_name}/metrics_results.pkl", 'rb') as file:
            self.metrics_results = pickle.load(file)
