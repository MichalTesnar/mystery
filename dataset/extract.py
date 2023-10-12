"""
Extracting the training and testing data.
"""

# Imports
import pandas as pd

class Dataset():
    csv_file_path = "dagon_dataset.csv"
    df = pd.read_csv(csv_file_path, skiprows=1)
    X = df.iloc[:, 0:6]
    y = df.iloc[:, 6:]
    TRAIN_SET_SIZE = 0.6
    VAL_SET_SIZE = 0.2
    TRAIN_SET_ITEMS = int(TRAIN_SET_SIZE*X.shape[0])
    VAL_SET_ITEMS = int(VAL_SET_SIZE*X.shape[0])

    @staticmethod
    def get_training_set():
        csv_file_path = "dagon_dataset.csv"
        df = pd.read_csv(csv_file_path, skiprows=1)
        X = df.iloc[:, 0:6]
        y = df.iloc[:, 6:]
        TRAIN_SET_SIZE = 0.6
        TRAIN_SET_ITEMS = int(TRAIN_SET_SIZE*X.shape[0])
        X_train, y_train = X[:TRAIN_SET_ITEMS], y[:TRAIN_SET_ITEMS]
        return X_train, y_train
    
    @staticmethod
    def get_validation_set():
        csv_file_path = "dagon_dataset.csv"
        df = pd.read_csv(csv_file_path, skiprows=1)
        X = df.iloc[:, 0:6]
        y = df.iloc[:, 6:]
        TRAIN_SET_SIZE = 0.6
        VAL_SET_SIZE = 0.2
        TRAIN_SET_ITEMS = int(TRAIN_SET_SIZE*X.shape[0])
        VAL_SET_ITEMS = int(VAL_SET_SIZE*X.shape[0])
        X_val, y_val = X[TRAIN_SET_ITEMS:TRAIN_SET_ITEMS + VAL_SET_ITEMS], y[TRAIN_SET_ITEMS:TRAIN_SET_ITEMS + VAL_SET_ITEMS]
        return X_val, y_val
    
    @staticmethod
    def get_test_set():
        csv_file_path = "dagon_dataset.csv"
        df = pd.read_csv(csv_file_path, skiprows=1)
        X = df.iloc[:, 0:6]
        y = df.iloc[:, 6:]
        TRAIN_SET_SIZE = 0.6
        VAL_SET_SIZE = 0.2
        TRAIN_SET_ITEMS = int(TRAIN_SET_SIZE*X.shape[0])
        VAL_SET_ITEMS = int(VAL_SET_SIZE*X.shape[0])
        X_test, y_test = X[TRAIN_SET_ITEMS + VAL_SET_ITEMS:], y[TRAIN_SET_ITEMS + VAL_SET_ITEMS:]
        return X_test, y_test
