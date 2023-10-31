class Metrics():
    def __init__(self, test_set) -> None:
        self._test_X, self._test_y = test_set
        self.metrics_results = {"MSE": [],
                                "R2:": []}

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
        print(self.metrics_results["MSE"])

    def save(self):
        """
        Save collected data.
        """
        pass