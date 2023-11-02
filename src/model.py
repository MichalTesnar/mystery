import numpy as np
import pandas as pd

class AIOModel():
    def __init__(self, training_set, mode="FIFO", p=0.5) -> None:
        assert mode in ["FIFO", "FIRO", "RIRO", "SPACE_HEURISTIC", "TIME_HEURISTIC"]
        self.mode = mode
        self.X_train, self.y_train = training_set
        self.p=p

    @property
    def get_mode(self):
        return self.mode

    def update_own_training_set(self, new_point):
        """"
        Update the training set with the new incoming points. Return false, as long as you did not update, then you keep being fed a new item.
        """

        new_X, new_y = new_point
        ################## BASELINES ##################
        if self.mode == "FIFO":
            self.X_train = self.X_train.iloc[1:]
            self.y_train = self.y_train.iloc[1:]
            self.X_train = pd.concat([self.X_train, new_X], ignore_index=True)
            self.y_train = pd.concat([self.y_train, new_y], ignore_index=True)
            return True
        
        if self.mode == "FIRO": ## First in, random out
            random_row_index = self.X_train.sample().index
            self.X_train = self.X_train.drop(random_row_index)
            self.y_train = self.y_train.drop(random_row_index)
            self.X_train = pd.concat([self.X_train, new_X], ignore_index=True)
            self.y_train = pd.concat([self.y_train, new_y], ignore_index=True)
            df = df.sample(frac=1) # shuffle
            return True
        
        if self.mode == "RIRO": ## Random in, random out
            if np.random.rand() > self.p: # Only accept with probability P
                return False
            random_row_index = self.X_train.sample().index
            self.X_train = self.X_train.drop(random_row_index)
            self.y_train = self.y_train.drop(random_row_index)
            self.X_train = pd.concat([self.X_train, new_X], ignore_index=True)
            self.y_train = pd.concat([self.y_train, new_y], ignore_index=True)
            df = df.sample(frac=1) # shuffle
            return True
        
        ################## HEURISTICS ##################
        if self.mode == "SPACE_HEURISTIC":
            # @TODO
            self.X_train = self.X_train.iloc[1:]
            self.y_train = self.y_train.iloc[1:]
            self.X_train = pd.concat([self.X_train, new_X], ignore_index=True)
            self.y_train = pd.concat([self.y_train, new_y], ignore_index=True)
            return True
        
        if self.mode == "TIME_HEURISTIC":
            # @TODO
            self.X_train = self.X_train.iloc[1:]
            self.y_train = self.y_train.iloc[1:]
            self.X_train = pd.concat([self.X_train, new_X], ignore_index=True)
            self.y_train = pd.concat([self.y_train, new_y], ignore_index=True)
            return True
        ################## UQ METHODS ##################



    def retrain(self):
        """
        Retrain yourself give the own dataset you have.
        """
        pass

    def predict(self, points):
        """
        Predict on the given set of points, also output uncertainty.
        """
        pass