import pickle
import numpy as np
import matplotlib.pyplot as plt

"""
@TODO: based on the implementation of .predict implement calculation of metrics
"""

class Metrics():
    def __init__(self, identifier, test_set) -> None:
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
        plt.savefig(f"{self.identifier}")
        plt.show()
        plt.close()

    def save(self):
        """
        Save the data collected in the dictionary.
        """
        with open(f"{self.identifier}_metrics_results.pkl", 'wb') as file:
            pickle.dump(self.metrics_results, file)

    def load(self):
        """
        Reload previously collected metrics from a file.
        """
        with open(f"{self.identifier}_metrics_results.pkl", 'rb') as file:
            self.metrics_results = pickle.load(file)
