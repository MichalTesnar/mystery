
class AIOModel():
    def __init__(self, training_set, mode="FIFO") -> None:
        self.mode = mode
        self.X_train, self.y_train = training_set
        # keep own training set
        # keep own learner
        pass

    @property
    def get_mode(self):
        return self.mode

    def update_own_training_set(self, new_point):
        """"
        Update the training set with the new incoming points. Return false, as long as you did not update, then you keep being fed a new item.
        """
        new_X, new_y = new_point
        return True

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