# Native Keras
from keras.layers import Input, Dense
from keras.models import Model
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
# UQ
from keras_uncertainty.models import SimpleEnsemble
# Tuner
from keras_tuner import BayesianOptimization, HyperModel
# My stuff
from src.dataset import SinusiodToyExample, DagonAUVDataset
from src.metrics import MetricsTuning
from src.model import AIOModelTuning
# Utils
import copy

es = {
    "EXPERIMENT_IDENTIFIER": "Debug7",
    "BUFFER_SIZE": 10,
    "MODEL_MODE": "THRESHOLD",
    "DATASET_MODE": "subsampled_sequential",
    "NUMBER_OF_LAYERS": 2,
    "UNITS_PER_LAYER": 2,
    "DATASET_SIZE": 0.005,
    "LEARNING_RATE": 0.01,
    "BATCH_SIZE": 2,
    "PATIENCE": 3,
    "MAX_EPOCHS": 20,
    "ACCEPT_PROBABILITY": 0.7,
    "INPUT_LAYER_SIZE": 1,
    "OUTPUT_LAYER_SIZE": 1,
    "UNCERTAINTY_THRESHOLD": 0.1,
    "RUNNING_MEAN_WINDOW": 10
}

dataset = SinusiodToyExample(experiment_specification=es)
X_train, y_train = dataset.give_initial_training_set(
    es["BUFFER_SIZE"])
X_val, y_val = dataset.get_validation_set


class MyHyperModel(HyperModel):
    def build(self, hp):
        def model_fn():
            # Input layer
            inp = Input(es["INPUT_LAYER_SIZE"])
            hp_units = hp.Int('units', min_value=1, max_value=4, step=1)
            x = Dense(hp_units, activation="relu")(inp)
            # Hyperparam tune number of layers
            num_layers = hp.Int('num_layers', min_value=1, max_value=4, step=1)
            for _ in range(num_layers):
                # for each add units
                x = Dense(hp_units, activation="relu")(x)
            mean = Dense(
                es["OUTPUT_LAYER_SIZE"], activation="linear")(x)
            train_model = Model(inp, mean)
            hp_learning_rate = hp.Choice('learning_rate', values=[
                                         1e-1, 1e-2, 1e-3, 1e-4, 1e-5])
            train_model.compile(loss="mse", optimizer=Adam(
                learning_rate=hp_learning_rate))
            return train_model

        return SimpleEnsemble(model_fn, num_estimators=10)
        # HAD TO ALTER KERAS BACKEND TO BE ABLE TO DO THIS, not a valid Keras Model

    def fit(self, hp, model, x, y, validation_data, callbacks=None, **kwargs):
        # hyperparams
        batch_size = hp.Int("batch_size", 1, 16, step=2, default=4)
        es["BATCH_SIZE"] = batch_size
        patience = hp.Int('patience', min_value=2, max_value=10)
        es["PATIENCE"] = patience
        buffer_size = hp.Int('buffer_size', min_value=2, max_value=10)
        es["PATIENCE"] = buffer_size
        # buffer size, other params

        current_dataset = copy.deepcopy(dataset)
        AIOmodel = AIOModelTuning(current_dataset.give_initial_training_set(
            es["BUFFER_SIZE"]), es, model)
        metrics = MetricsTuning(current_dataset.get_current_training_set_size,  # account for extra iteration at the end
                                es, dataset.get_validation_set)

        training_flag = True
        while current_dataset.data_available():
            if training_flag:
                AIOmodel.retrain(verbose=False)
                metrics.collect_metrics(AIOmodel)
            training_flag = False
            while not training_flag and current_dataset.data_available():
                new_point = current_dataset.get_new_point()
                metrics.collect_uncertainty(AIOmodel, new_point)
                training_flag = AIOmodel.update_own_training_set(new_point)
                if not training_flag and current_dataset.data_available():
                    metrics.pad_metrics()

        # fit on cummulative MSE loss
        return metrics.metrics_results["MSE"][-1]


tuner = BayesianOptimization(hypermodel=MyHyperModel(),
                             objective='val_loss',
                             directory='hyperparams',
                             project_name=es["EXPERIMENT_IDENTIFIER"])

tuner.search(None, None, epochs=es["MAX_EPOCHS"], validation_data=(None, None))

# Get the optimal hyperparameters
best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
print(f"""
number of layers {best_hps.get('num_layers')}.
units per layer {best_hps.get('units')}
learning rate {best_hps.get('learning_rate')}.
batch size {best_hps.get('batch_size')}.
patience {best_hps.get('patience')}.
""")
