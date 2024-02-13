import pandas as pd
import numpy as np
from math import sin, exp
import time

class Dataset():
    def __init__(self, experiment_specification, normalize=False):
        assert experiment_specification["DATASET_SIZE"] > 0 and experiment_specification["DATASET_SIZE"] <= 1, "Choose valid (sub)set size."
        assert experiment_specification["DATASET_MODE"] in ["random", "sequential",
                        "subsampled_sequential"], "Dataset mode not implemented."
        self._load_data(experiment_specification["DATASET_SIZE"], normalize=normalize)
        
        self._X = self._df.iloc[:, 0:experiment_specification["INPUT_LAYER_SIZE"]]
        self._y = self._df.iloc[:, experiment_specification["INPUT_LAYER_SIZE"]:]
        self._TRAIN_SET_SIZE = 0.6
        self._VAL_SET_SIZE = 0.2
        self._TEST_SET_SIZE = 1 - self._TRAIN_SET_SIZE - self._VAL_SET_SIZE
        self._TRAIN_SET_ITEMS = int(self._TRAIN_SET_SIZE*self._X.shape[0])
        self._VAL_SET_ITEMS = int(self._VAL_SET_SIZE*self._X.shape[0])
        self._TEST_SET_ITEMS = int(self._TEST_SET_SIZE*self._X.shape[0])
        # assign the sets
        self._get_sets(experiment_specification["DATASET_MODE"])
        self._X_point_queue = self._X_train.copy()
        self._y_point_queue = self._y_train.copy()
        self.initial_number_of_points = self._X_point_queue.shape[0]

    def _get_sets(self, mode):
        """
        @param mode: Selected mode of creation of dataset.
            "random"                : use random subset of the data.
            "sequential"            : take last part (based on time) of the dataset as test
            "subsampled_sequential" : sample the set at equal time stamps throught out the process to obtainthe test set
        """
        if mode == "random":
            self._df = self._df.sample(frac=1)
            self._X_train, self._y_train = self._X[:
                                                   self._TRAIN_SET_ITEMS], self._y[:self._TRAIN_SET_ITEMS]
            self._X_val, self._y_val = self._X[self._TRAIN_SET_ITEMS:self._TRAIN_SET_ITEMS +
                                               self._VAL_SET_ITEMS], self._y[self._TRAIN_SET_ITEMS:self._TRAIN_SET_ITEMS + self._VAL_SET_ITEMS]
            self._X_test, self._y_test = self._X[self._TRAIN_SET_ITEMS +
                                                 self._VAL_SET_ITEMS:], self._y[self._TRAIN_SET_ITEMS + self._VAL_SET_ITEMS:]
        elif mode == "subsampled_sequential":
            freq_test = int(self._X.shape[0]/self._TEST_SET_ITEMS)
            indices_test = np.arange(1, self._X.shape[0], freq_test)
            self._X_test, self._y_test = self._X.iloc[indices_test].reset_index(
                drop=True), self._y.iloc[indices_test].reset_index(drop=True)
            X_train_val, y_train_val = self._X.drop(indices_test).reset_index(
                drop=True), self._y.drop(indices_test).reset_index(drop=True)
            ratio = self._VAL_SET_ITEMS / \
                (self._VAL_SET_ITEMS + self._TRAIN_SET_ITEMS)
            freq_val = int(1/ratio)
            indices_val = np.arange(1, X_train_val.shape[0], freq_val)
            self._X_val, self._y_val = X_train_val.iloc[indices_val].reset_index(
                drop=True), y_train_val.iloc[indices_val].reset_index(drop=True)
            self._X_train, self._y_train = X_train_val.drop(indices_val).reset_index(
                drop=True), y_train_val.drop(indices_val).reset_index(drop=True)
        elif mode == "sequential":
            self._X_train, self._y_train = self._X[:
                                                   self._TRAIN_SET_ITEMS], self._y[:self._TRAIN_SET_ITEMS]
            self._X_val, self._y_val = self._X[self._TRAIN_SET_ITEMS:self._TRAIN_SET_ITEMS +
                                               self._VAL_SET_ITEMS], self._y[self._TRAIN_SET_ITEMS:self._TRAIN_SET_ITEMS + self._VAL_SET_ITEMS]
            self._X_test, self._y_test = self._X[self._TRAIN_SET_ITEMS +
                                                 self._VAL_SET_ITEMS:], self._y[self._TRAIN_SET_ITEMS + self._VAL_SET_ITEMS:]

    @property
    def get_training_set(self):
        return self._X_train.values, self._y_train.values
    
    @property
    def get_current_training_set_size(self):
        return self._X_point_queue.shape[0]
    
    @property
    def get_validation_set(self):
        return self._X_val.values, self._y_val.values
    
    @property
    def get_test_set(self):
        return self._X_test.values, self._y_test.values

    @property
    def get_remaining_data(self) -> int:
        """
        Checks if there is data available in the queue that we can train on.
        """
        return len(self._X_point_queue)

    def data_available(self, verbose=False, start_time=0) -> bool:
        """
        Checks if there is data available in the queue that we can train on.
        """        
        if verbose:
            print(f"Time {time.time()-start_time:.2f}: used {(self.initial_number_of_points - self.get_remaining_data)/self.initial_number_of_points*100:.2f}% of the points.")
        return len(self._X_point_queue) != 0

    def get_new_point(self):
        """
        Pop a point from the available data queue, shorten the available dataset.
        @return (X, y) pair of the next training point (None, if it does not exist)
        """
        if not self.data_available():
            return None
        first_X = self._X_point_queue.iloc[0].copy()
        first_y = self._y_point_queue.iloc[0].copy()
        self._X_point_queue = self._X_point_queue.iloc[1:]
        self._y_point_queue = self._y_point_queue.iloc[1:]
        return first_X.values, first_y.values

    def give_initial_training_set(self, number_of_points):
        """
        Pop more points from the queue at the start to give model some training set.
        @return X, y which are pairs training set
        """
        if not self.data_available():
            return None
        X_train = self._X_point_queue.head(number_of_points)
        y_train = self._y_point_queue.head(number_of_points)

        self._X_point_queue = self._X_point_queue.drop(self._X_point_queue.index[:number_of_points])
        self._y_point_queue = self._y_point_queue.drop(self._y_point_queue.index[:number_of_points])
        return X_train.values, y_train.values
    

class DagonAUVDataset(Dataset):
    def _load_data(self, size, normalize=False):
        """
        Load the data from the storage.
        """
        file_path = "dagon_dataset.csv"
        to_read = int(size*11566)
        self._df = pd.read_csv(file_path, skiprows=1, nrows=to_read)
        if normalize:
            # Normalize each numeric column to the range [0, 1]
            numeric_columns = self._df.select_dtypes(include=['float64', 'int64']).columns
            for column in numeric_columns:
                min_value = self._df[column].min()
                max_value = self._df[column].max()
                self._df[column] = (self._df[column] - min_value) / (max_value - min_value)
            
            # save to csv
        # self._df.to_csv("dagon_dataset_normalized.csv", index=False)
        # exit()

class SinusiodToyExample(Dataset):
    def _load_data(self, size):
        """
        Load the data from the storage.
        """
        sample_rate = int(10000*size)
        domain = np.linspace(-6, 6, num=sample_rate)
        # np.random.shuffle(domain)
        domain_y = self.toy_function(domain)
        self._df = pd.DataFrame({'x': domain, 'y': domain_y})
        

    def toy_function(self, input):
        """
        Generating sinusiod training data with added noise.
        """
        output = []
        for inp in input:
            std = max(0.03 / (1.0 + exp(-inp)), 0)
            out = sin(inp) #+ np.random.normal(0, std) + inp*np.random.normal(0, std)
            output.append(5*out)
        return np.array(output)
