import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF

class BigGRP():
    def __init__(self, input_dim) -> None:
        self._kernel = RBF(length_scale = [5] *input_dim, length_scale_bounds=(1e-5, 1e5))
        self._model = GaussianProcessRegressor(kernel=self._kernel, n_restarts_optimizer=100)

    def retrain(self, x_train, y_train):
        self._model.fit(x_train, y_train)

    def predict(self, points):
        pred_mean, pred_std = self._model.predict(points, return_std=True)
        return pred_mean, pred_std
    
    def loss(self, x_test, y_test): # pure MAE without looking at variance
        pred_mean, _ = self.predict(x_test)
        return np.sum(np.sum(np.abs(pred_mean-y_test)))


def test():
    input_dim = 6
    grp = BigGRP(6)

    # Generate example data
    x_train = np.random.rand(200, input_dim)  # Replace this with your actual training data
    a = np.sin(np.sum(x_train, axis=1))
    b = np.sum(x_train, axis=1)
    c = np.exp(np.sum(x_train, axis=1))
    y_train = np.stack((a, b, c)).T
    # Retrain the model
    grp.retrain(x_train, y_train)

    # Make predictions
    test_point = np.random.rand(1, input_dim)  # Replace this with your actual test data
    prediction_mean, prediction_std = grp.predict(test_point)

    print("Prediction True:", np.sin(np.sum(test_point, axis=1)), np.sum(test_point, axis=1), np.exp(np.sum(test_point, axis=1)))
    print("Prediction Mean:", prediction_mean)
    print("Prediction Standard Deviation:", prediction_std)
