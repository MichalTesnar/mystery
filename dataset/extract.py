"""
Extracting the training and testing data.
"""

# Imports
import pandas as pd

class Dataset():
    def __init__(self):
        self._csv_file_path = "dagon_dataset.csv"
        self._df = pd.read_csv(self._csv_file_path, skiprows=1)
        self._X = self._df.iloc[:, 0:6]
        self._y = self._df.iloc[:, 6:]
        self._TRAIN_SET_SIZE = 0.6
        self._VAL_SET_SIZE = 0.2
        self._TRAIN_SET_ITEMS = int(self._TRAIN_SET_SIZE*self._X.shape[0])
        self._VAL_SET_ITEMS = int(self._VAL_SET_SIZE*self._X.shape[0])

    def get_training_set(self):
        X_train, y_train = self._X[:self._TRAIN_SET_ITEMS], self._y[:self._TRAIN_SET_ITEMS]
        return X_train, y_train
    
    def get_validation_set(self):
        X_val, y_val = self._X[self._TRAIN_SET_ITEMS:self._TRAIN_SET_ITEMS + self._VAL_SET_ITEMS], self._y[self._TRAIN_SET_ITEMS:self._TRAIN_SET_ITEMS + self._VAL_SET_ITEMS]
        return X_val, y_val
    
    def get_test_set(self):
        X_test, y_test = self._X[self._TRAIN_SET_ITEMS + self._VAL_SET_ITEMS:], self._y[self._TRAIN_SET_ITEMS + self._VAL_SET_ITEMS:]
        return X_test, y_test
