# Native Keras
from keras.layers import Input, Dense
from keras.models import Model
from keras.optimizers import Adam
# UQ
from keras_uncertainty.models import SimpleEnsemble
# Tuner
from keras_tuner import BayesianOptimization, HyperModel
# My stuff
from src.dataset import SinusiodToyExample, DagonAUVDataset
from src.metrics import MetricsTuning, Metrics
from src.model import AIOModelTuning, AIOModel
# Utils
import copy
import sys
import numpy as np

np.random.seed(107)

DATASET_TYPE = "Dagon"
MODEL_MODE = sys.argv[1]

es = {
    "EXPERIMENT_IDENTIFIER": f"Full data fix IT {MODEL_MODE}",
    "EXPERIMENT_TYPE": DATASET_TYPE,
    "BUFFER_SIZE": 100,
    "MODEL_MODE": MODEL_MODE,
    "DATASET_MODE": "subsampled_sequential",
    "DATASET_SIZE": 0.02,
    "MAX_EPOCHS": 100 if MODEL_MODE != "OFFLINE" else 100*7000,
    "ACCEPT_PROBABILITY": 0.7,
    "INPUT_LAYER_SIZE": 6 if DATASET_TYPE == "Dagon" else 1,
    "OUTPUT_LAYER_SIZE": 3 if DATASET_TYPE == "Dagon" else 1,
    "UNCERTAINTY_THRESHOLD": 0.02,
    "NUMBER_OF_ESTIMATORS": 10
}

if es["EXPERIMENT_TYPE"] == "Dagon":
    dataset = DagonAUVDataset(es)
elif es["EXPERIMENT_TYPE"] == "Toy":
    dataset = SinusiodToyExample(es)

class MyHyperModel(HyperModel):
    def build(self, hp):
        # model building function
        def model_fn():
            inp = Input(es["INPUT_LAYER_SIZE"])
            hp_units = hp.Choice('units', values=[4, 8, 16, 32, 64])
            x = Dense(hp_units, activation="relu")(inp)
            num_layers = hp.Int('num_layers', min_value=1, max_value=4, step=1)
            for _ in range(num_layers): # for each add units
                x = Dense(hp_units, activation="relu")(x)
            mean = Dense(es["OUTPUT_LAYER_SIZE"], activation="linear")(x)
            train_model = Model(inp, mean)
            hp_learning_rate = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4, 1e-5])
            train_model.compile(loss="mse", optimizer=Adam(learning_rate=hp_learning_rate))
            batch_size = hp.Choice('batch_size', values=[1, 2, 4, 8, 16])
            es["BATCH_SIZE"] = batch_size
            patience = hp.Choice('patience', values=[3, 5, 9])
            es["PATIENCE"] = patience
            return train_model

        return SimpleEnsemble(model_fn, num_estimators=es["NUMBER_OF_ESTIMATORS"])
        # HAD TO ALTER KERAS BACKEND TO BE ABLE TO DO THIS, not a valid Keras Model

    def fit(self, hp, model, x, y, validation_data, callbacks=None, **kwargs):
        current_dataset = copy.deepcopy(dataset)
        if MODEL_MODE != "OFFLINE":
            AIOmodel = AIOModelTuning(current_dataset.give_initial_training_set(es["BUFFER_SIZE"]), es, model)
            metrics = MetricsTuning(current_dataset.get_current_training_set_size, es, current_dataset.get_validation_set)
            whole = current_dataset.get_current_training_set_size
            i = 0
            training_flag = True
            while current_dataset.data_available():
                print(f"Point:{i}/{whole}")
                if training_flag:
                    AIOmodel.retrain(verbose=False)
                    metrics.collect_metrics(AIOmodel)
                    if DATASET_TYPE == "Toy":
                        metrics.extra_plots(AIOmodel)
                training_flag = False
                while not training_flag and current_dataset.data_available():
                    new_point = current_dataset.get_new_point()
                    training_flag = AIOmodel.update_own_training_set(new_point)
                    i += 1
                    if not training_flag and current_dataset.data_available():
                        metrics.pad_metrics()
            # fit on cummulative MSE loss
            return metrics.metrics_results["Cummulative MSE"][-1]
        else:
            AIOmodel = AIOModelTuning(dataset.get_training_set, es, model)
            metrics = MetricsTuning(1, es, dataset.get_validation_set)
            AIOmodel.retrain()
            metrics.collect_metrics(model)
            return metrics.metrics_results["MSE"][-1]


tuner = BayesianOptimization(hypermodel=MyHyperModel(),
                             objective='val_loss',
                             directory='hyperparams',
                             project_name=es["EXPERIMENT_IDENTIFIER"],
                             max_trials=60)
tuner.search(None, None, epochs=es["MAX_EPOCHS"], validation_data=(None, None))
