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
    def __init__(self, iterations=0, experiment_specification={}, test_set=([],[]), load=False) -> None:
        self.identifier = experiment_specification["EXPERIMENT_IDENTIFIER"]
        if load:
            self.dir_name = f"results/{self.identifier}"
            print(f"Loading the results saved in {self.dir_name}")
            self.load()

        else:
            dir_i = 0
            while os.path.isdir(f"results/{self.identifier} ({dir_i})"):
                dir_i += 1
            self.dir_name = f"results/{self.identifier} ({dir_i})"
            os.mkdir(self.dir_name)
            print(f"The results will be saved in {self.dir_name}")
            self.metrics_results = {"MSE": np.zeros(iterations),
                                    "R2": np.zeros(iterations),
                                    "Cummulative MSE": np.zeros(iterations),
                                    "Skips": np.zeros(iterations)}

        self.current_data_index = 0
        self.model_specification = experiment_specification

        self._test_X, self._test_y = test_set
        self.test_set_size = len(self._test_X)

    def collect_metrics(self, model):
        """
        Apply the metrics and save the results.
        """
        pred_mean, pred_std = model.predict(self._test_X)
        for metric in self.metrics_results.keys():
            self.metrics_results[metric][self.current_data_index] = self.calculate_metric(
                metric, pred_mean, self.metrics_results[metric][self.current_data_index-1])
        self.current_data_index += 1

    def pad_metrics(self):
        """
        Repeat last values in the array if you have skipped an iteration.
        """
        for metric in self.metrics_results.keys():
            if metric == "Skips":
                self.metrics_results[metric][self.current_data_index] = self.metrics_results[metric][self.current_data_index - 1] + 1
            else:
                self.metrics_results[metric][self.current_data_index] = self.metrics_results[metric][self.current_data_index - 1]
        self.current_data_index += 1
    
    def restore_cummulativeMSE(self):
        last_value = 0
        for i in range(len(self.metrics_results["MSE"])):
            self.metrics_results["Cummulative MSE"][i] = self.metrics_results["MSE"][i] + last_value
            last_value = self.metrics_results["Cummulative MSE"][i]


    def calculate_metric(self, key, pred_mean, last_value):
        """
        Calculates the individual metrics.
        """
        if key == "MSE":
            return np.sum(np.square(self._test_y - pred_mean))/self.test_set_size
        if key == "R2":
            return r2_score(pred_mean, self._test_y)
        if key == "Cummulative MSE":
            return last_value + np.sum(np.square(self._test_y - pred_mean))/self.test_set_size
        if key == "Skips":
            return last_value

    def plot(self):
        """
        Plot your metrics.
        """
        fig, axs = plt.subplots(
            len(self.metrics_results.keys()), 1, figsize=(16, 12))
        fig.suptitle(f"Online Learning Metrics for {self.identifier}", fontsize=20)
        for i, metric in enumerate(self.metrics_results.keys()):
            y = self.metrics_results[metric]
            x = np.arange(0, len(y))
            axs[i].plot(x, y, label=metric)
            axs[i].set_title(metric)
            axs[i].legend(loc='upper left', fontsize=20)

        plt.tight_layout()
        plt.savefig(f"{self.dir_name}/plotted_metrics")
        # plt.show()
        plt.close()

    def extra_plots(self, model):
        """
        Convenience plotting to show incremental learning in the toy example.
        """
        y_pred_mean, y_pred_std = model.predict(self._test_X)
        y_pred_mean = np.array(y_pred_mean)
        y_pred_mean = y_pred_mean.reshape((-1,))
        y_pred_std = np.array(y_pred_std)
        y_pred_std = y_pred_std.reshape((-1,))
        y_pred_up_1 = y_pred_mean + y_pred_std
        y_pred_down_1 = y_pred_mean - y_pred_std
        fig, ax = plt.subplots()
        ax.set_title(f"Plotting Iterations of {self.identifier}")
        ax.set_ylim([-20.0, 20.0])
        ax.plot(model.X_train, model.y_train, '.', color=(
            0.9, 0, 0, 0.5), markersize=15, label="training set")
        ax.plot(self._test_X, self._test_y, '.', color=(0, 0.9, 0, 1),
                markersize=3, label="ground truth")
        ax.fill_between(self._test_X.ravel(), y_pred_down_1,
                        y_pred_up_1,  color=(0, 0.5, 0.9, 0.5))
        ax.plot(self._test_X.ravel(), y_pred_mean, '.',
                color=(1, 1, 1, 0.8), markersize=0.2)
        ax.legend(bbox_to_anchor=(1, 1))
        plt.savefig(f"{self.dir_name}/iteration {self.current_data_index}")
        plt.close()

    def save(self):
        """
        Save the data collected in the dictionary.
        """

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
