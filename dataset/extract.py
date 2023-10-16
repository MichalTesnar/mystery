"""
Extracting the training and testing data.
"""

# Imports
import pandas as pd
import numpy as np

class Dataset():
    def __init__(self, mode="random", size=1):
        assert size > 0 and size <= 1, "Choose valid set size."
        assert mode in ["random", "sequential", "subsumpled_sequential"], "Mode not implemented."

        self._csv_file_path = "dagon_dataset.csv"
        SIZE = 11566
        to_read = int(size*SIZE)
        self._df = pd.read_csv(self._csv_file_path, skiprows=1, nrows=to_read)
        self._X = self._df.iloc[:, 0:6]
        self._y = self._df.iloc[:, 6:]
        self._TRAIN_SET_SIZE = 0.6
        self._VAL_SET_SIZE = 0.2
        self._TEST_SET_SIZE = 1 - self._TRAIN_SET_SIZE - self._VAL_SET_SIZE 
        self._TRAIN_SET_ITEMS = int(self._TRAIN_SET_SIZE*self._X.shape[0])
        self._VAL_SET_ITEMS = int(self._VAL_SET_SIZE*self._X.shape[0])
        self._TEST_SET_ITEMS = int(self._TEST_SET_SIZE*self._X.shape[0])
        # assign the sets
        self._get_sets(mode)

    def _get_sets(self, mode):
        if mode == "random":
            self._df = self._df.sample(frac=1)
            self._X_train, self._y_train = self._X[:self._TRAIN_SET_ITEMS], self._y[:self._TRAIN_SET_ITEMS]
            self._X_val, self._y_val = self._X[self._TRAIN_SET_ITEMS:self._TRAIN_SET_ITEMS + self._VAL_SET_ITEMS], self._y[self._TRAIN_SET_ITEMS:self._TRAIN_SET_ITEMS + self._VAL_SET_ITEMS]
            self._X_test, self._y_test = self._X[self._TRAIN_SET_ITEMS + self._VAL_SET_ITEMS:], self._y[self._TRAIN_SET_ITEMS + self._VAL_SET_ITEMS:]
        elif mode == "subsumpled_sequential":
            freq_test = int(self._X.shape[0]/self._TEST_SET_ITEMS)
            indices_test = np.arange(1,self._X.shape[0], freq_test)
            self._X_test, self._y_test = self._X.iloc[indices_test].reset_index(drop=True), self._y.iloc[indices_test].reset_index(drop=True)
            X_train_val, y_train_val = self._X.drop(indices_test).reset_index(drop=True), self._y.drop(indices_test).reset_index(drop=True)
            ratio = self._VAL_SET_ITEMS/(self._VAL_SET_ITEMS + self._TRAIN_SET_ITEMS)
            freq_val = int(1/ratio)
            indices_val = np.arange(1,X_train_val.shape[0], freq_val)
            self._X_val, self._y_val = X_train_val.iloc[indices_val].reset_index(drop=True), y_train_val.iloc[indices_val].reset_index(drop=True)
            self._X_train, self._y_train = X_train_val.drop(indices_val).reset_index(drop=True), y_train_val.drop(indices_val).reset_index(drop=True)
        elif mode == "sequential":
            self._X_train, self._y_train = self._X[:self._TRAIN_SET_ITEMS], self._y[:self._TRAIN_SET_ITEMS]
            self._X_val, self._y_val = self._X[self._TRAIN_SET_ITEMS:self._TRAIN_SET_ITEMS + self._VAL_SET_ITEMS], self._y[self._TRAIN_SET_ITEMS:self._TRAIN_SET_ITEMS + self._VAL_SET_ITEMS]
            self._X_test, self._y_test = self._X[self._TRAIN_SET_ITEMS + self._VAL_SET_ITEMS:], self._y[self._TRAIN_SET_ITEMS + self._VAL_SET_ITEMS:]
        
        self._X_train.columns = ['u', 'v', 'r', 'th1', 'th2', 'th3']
        self._X_val.columns = ['u', 'v', 'r', 'th1', 'th2', 'th3']
        self._X_test.columns = ['u', 'v', 'r', 'th1', 'th2', 'th3']
        self._y_train.columns = ['du', 'dv', 'dr']
        self._y_val.columns = ['du', 'dv', 'dr']
        self._y_test.columns = ['du', 'dv', 'dr']


    def get_training_set(self):
        return self._X_train, self._y_train
    
    def get_validation_set(self):
        return self._X_val, self._y_val
    
    def get_test_set(self):
        return self._X_test, self._y_test
