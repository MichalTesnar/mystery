import numpy as np
import pandas as pd

from keras.models import Model
from keras.layers import Input, Dense
from keras.callbacks import EarlyStopping
from keras_uncertainty.models import SimpleEnsemble


class AIOModel():
    def __init__(self, training_set, experiment_specification, p=0.5) -> None:
        assert experiment_specification["MODEL_MODE"] in ["FIFO", "FIRO", "RIRO", "SPACE_HEURISTIC",
                                                          "TIME_HEURISTIC", "GREEDY", "THRESHOLD", "THRESHOLD_GREEDY"], "Mode does not exist."
        self.experiment_specification = experiment_specification
        self.X_train, self.y_train = training_set
        self.construct_ensembles()

    def construct_ensembles(self):
        def model_fn():
            inp = Input(shape=(self.experiment_specification["INPUT_LAYER_SIZE"],))
            x = Dense(128, activation="relu")(inp)
            x = Dense(128, activation="relu")(x)
            x = Dense(64, activation="relu")(x)
            x = Dense(32, activation="relu")(x)
            mean = Dense(self.experiment_specification["OUTPUT_LAYER_SIZE"], activation="linear")(x)
            train_model = Model(inp, mean)
            train_model.compile(loss="mse", optimizer="adam")
            return train_model

        self.model = SimpleEnsemble(model_fn, num_estimators=10)

    def update_own_training_set(self, new_point):
        """"
        Update the training set with the new incoming points. Return false, as long as you did not update, then you keep being fed a new item.
        """

        new_X, new_y = new_point

        ################## BASELINES ##################
        if self.experiment_specification["MODEL_MODE"] == "FIFO":
            self.X_train = self.X_train[1:]
            self.y_train = self.y_train[1:]
            self.X_train = np.concatenate(
                [self.X_train, new_X.reshape(-1, 1).T])
            self.y_train = np.concatenate(
                [self.y_train, new_y.reshape(-1, 1).T])
            return True

        # First in, random out
        elif self.experiment_specification["MODEL_MODE"] == "FIRO":
            random_index = np.random.choice(self.X_train.shape[0])
            self.X_train[random_index] = new_X
            self.y_train[random_index] = new_y
            np.random.shuffle(self.X_train)
            np.random.shuffle(self.X_train)
            return True

        # Random in, random out
        elif self.experiment_specification["MODEL_MODE"] == "RIRO":
            # Only accept with probability 'p'
            if np.random.rand() > self.experiment_specification["ACCEPT_PROBABILITY"]:
                return False
            random_index = np.random.choice(self.X_train.shape[0])
            self.X_train[random_index] = new_X
            self.y_train[random_index] = new_y
            np.random.shuffle(self.X_train)
            np.random.shuffle(self.X_train)
            return True

        ################## HEURISTICS ##################
        elif self.experiment_specification["MODEL_MODE"] == "SPACE_HEURISTIC":
            # @TODO
            raise NotImplemented("This method is not implemented.")
            return True

        elif self.experiment_specification["MODEL_MODE"] == "TIME_HEURISTIC":
            # @TODO
            raise NotImplemented("This method is not implemented.")

        ################## UQ METHODS ##################
        elif self.experiment_specification["MODEL_MODE"] == "GREEDY":
            # @TODO
            raise NotImplemented("This method is not implemented.")

        elif self.experiment_specification["MODEL_MODE"] == "THRESHOLD":
            # @TODO
            raise NotImplemented("This method is not implemented.")

        elif self.experiment_specification["MODEL_MODE"] == "THRESHOLD_GREEDY":
            # @TODO
            raise NotImplemented("This method is not implemented.")

    def retrain(self):
        """
        Retrain yourself give the own dataset you have.
        """
        early_stop = EarlyStopping(
            monitor='loss', patience=self.experiment_specification["PATIENCE"])
        history = self.model.fit(self.X_train, self.y_train, verbose=False, epochs=self.experiment_specification["MAX_EPOCHS"], callbacks=[
                                 early_stop], batch_size=self.experiment_specification["BATCH_SIZE"])
        return history

    def predict(self, points):
        """
        Predict on the given set of points, also output uncertainty.
        """
        pred_mean, pred_std = self.model.predict(points)
        return pred_mean, pred_std
