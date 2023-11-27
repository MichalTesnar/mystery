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
import sys

DATASET_TYPE = "Dagon"
EXP_TYPE = "Online"
MODEL_MODE = sys.argv[1]
print("Model mode is", MODEL_MODE)

es = {
    "EXPERIMENT_IDENTIFIER": f"Dagon try {MODEL_MODE}",
    "EXPERIMENT_TYPE": DATASET_TYPE,
    "BUFFER_SIZE": 50,
    "MODEL_MODE": MODEL_MODE,
    "DATASET_MODE": "subsampled_sequential",
    "NUMBER_OF_LAYERS": 4,
    "UNITS_PER_LAYER": 32,
    "DATASET_SIZE": 0.05,
    "MAX_EPOCHS": 200,
    "ACCEPT_PROBABILITY": 0.7,
    "INPUT_LAYER_SIZE": 6 if DATASET_TYPE == "Dagon" else 1,
    "OUTPUT_LAYER_SIZE": 3 if DATASET_TYPE == "Dagon" else 1,
    "UNCERTAINTY_THRESHOLD": 0.1,
    "RUNNING_MEAN_WINDOW": 10,
    "NUMBER_OF_ESTIMATORS": 10
}

print("loading dataset")
if DATASET_TYPE == "Sinus":
    dataset = SinusiodToyExample(experiment_specification=es)
else:
    dataset = DagonAUVDataset(experiment_specification=es)
print("getting actual data")
X_train, y_train = dataset.give_initial_training_set(es["BUFFER_SIZE"])
X_val, y_val = dataset.get_validation_set

class MyHyperModel(HyperModel):
    def build(self, hp):
        # model building function
        def model_fn():
            inp = Input(es["INPUT_LAYER_SIZE"])
            hp_units = hp.Choice('units', values=[4, 8, 16, 32, 64])
            x = Dense(hp_units, activation="relu")(inp)
            # Hyperparam tune number of layers
            num_layers = hp.Int('num_layers', min_value=1, max_value=4, step=1)
            for _ in range(num_layers):
                # for each add units
                x = Dense(hp_units, activation="relu")(x)
            mean = Dense(es["OUTPUT_LAYER_SIZE"], activation="linear")(x)
            train_model = Model(inp, mean)
            hp_learning_rate = hp.Choice('learning_rate', values=[1e-1, 1e-2, 1e-3, 1e-4, 1e-5])
            train_model.compile(loss="mse", optimizer=Adam(learning_rate=hp_learning_rate))
            return train_model
        # lol

        return SimpleEnsemble(model_fn, num_estimators=es["NUMBER_OF_ESTIMATORS"])
        # HAD TO ALTER KERAS BACKEND TO BE ABLE TO DO THIS, not a valid Keras Model

    def fit(self, hp, model, x, y, validation_data, callbacks=None, **kwargs):
        print("Entering fitting")
	# hyperparams
        batch_size = hp.Choice('batch_size', values=[1, 2, 4, 8, 16])
        es["BATCH_SIZE"] = batch_size
        patience = hp.Int('patience', min_value=2, max_value=10)
        es["PATIENCE"] = patience

        current_dataset = copy.deepcopy(dataset)
        print("Copied trainig set")
        AIOmodel = AIOModelTuning(current_dataset.give_initial_training_set(es["BUFFER_SIZE"]), es, model)
        print("Compiled model")
        metrics = MetricsTuning(current_dataset.get_current_training_set_size, es, dataset.get_validation_set)
        print("Compiled metrics")
        whole = current_dataset.get_current_training_set_size
        i = 0
        training_flag = True
        while current_dataset.data_available():
            if training_flag:
                AIOmodel.retrain(verbose=False)
                print(f"DONE RETRAINING {i}/{whole}")
                i+=1
                metrics.collect_metrics(AIOmodel)
            training_flag = False
            while not training_flag and current_dataset.data_available():
                new_point = current_dataset.get_new_point()
                training_flag = AIOmodel.update_own_training_set(new_point)
                if not training_flag and current_dataset.data_available():
                    print("LOL OMG JUST SKIPEEd")
                    metrics.pad_metrics()
        # fit on cummulative MSE loss
        return metrics.metrics_results["MSE"][-1]


tuner = BayesianOptimization(hypermodel=MyHyperModel(),
                             objective='val_loss',
                             directory='hyperparams',
                             project_name=es["EXPERIMENT_IDENTIFIER"],
                             max_trials=20)
print("Starting the search")
tuner.search(None, None, epochs=es["MAX_EPOCHS"], validation_data=(None, None))

# Get the optimal hyperparameters
best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
