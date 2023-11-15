
import tensorflow as tf
from tensorflow import keras
from keras_tuner import Hyperband
from keras.layers import Input, Dense
from keras.optimizers import Adam
from keras.losses import MeanSquaredError
from src.dataset import SinusiodToyExample, DagonAUVDataset
from keras.callbacks import EarlyStopping

experiment_specification = {
    "EXPERIMENT_IDENTIFIER": "hp sinus with constrained size",
    "BUFFER_SIZE": 100,
    "MODEL_MODE": "FIRO",
    "DATASET_MODE": "subsampled_sequential",
    "DATASET_SIZE": 0.1,
    "BATCH_SIZE": 2,
    "PATIENCE": 10,
    "MAX_EPOCHS": 1000,
    "ACCEPT_PROBABILITY": 0.3,
    "INPUT_LAYER_SIZE": 1,
    "OUTPUT_LAYER_SIZE": 1,
    "UNCERTAINTY_THRESHOLD": 0.1
}

dataset = SinusiodToyExample(experiment_specification=experiment_specification)
# dataset = DagonAUVDataset(experiment_specification=experiment_specification)
X_train, y_train = dataset.give_initial_training_set(
    experiment_specification["BUFFER_SIZE"])


def model_builder(hp):
    model = keras.Sequential()
    model.add(Input(shape=(experiment_specification["INPUT_LAYER_SIZE"])))

    num_layers = hp.Int('num_layers', min_value=1, max_value=4, step=1)

    for _ in range(num_layers):
        hp_units = hp.Int('units', min_value=2, max_value=16, step=1)
        model.add(Dense(units=hp_units, activation='relu'))

    model.add(
        Dense(experiment_specification["OUTPUT_LAYER_SIZE"], activation="linear"))
    hp_learning_rate = hp.Choice('learning_rate', values=[1e-1, 1e-2, 1e-3, 1e-4, 1e-5])

    model.compile(optimizer=Adam(learning_rate=hp_learning_rate),
                  loss=MeanSquaredError())

    return model

# https://kegui.medium.com/how-to-tune-the-number-of-epochs-and-batch-size-in-keras-tuner-c2ab2d40878d


class MyTuner(Hyperband):
    def run_trial(self, trial, *args, **kwargs):
        kwargs['batch_size'] = trial.hyperparameters.Int(
            'batch_size', 1, 16, step=1)
        early_stopping = EarlyStopping(monitor='val_loss',
                                       patience=trial.hyperparameters.Int(
                                           'patience', min_value=5, max_value=20, step=1),
                                       restore_best_weights=True
                                       )
        if 'callbacks' in kwargs:
            kwargs['callbacks'].append(early_stopping)
        else:
            kwargs['callbacks'] = [early_stopping]
        return super(MyTuner, self).run_trial(trial, *args, **kwargs)


tuner = MyTuner(model_builder,
                objective='val_loss',
                directory='hyperparams',
                project_name=experiment_specification["EXPERIMENT_IDENTIFIER"])

# stop_early = EarlyStopping(monitor='val_loss', patience=5)
# , callbacks=[stop_early])
tuner.search(X_train, y_train, epochs=500, validation_split=0.3)

# Get the optimal hyperparameters
best_hps = tuner.get_best_hyperparameters(num_trials=1)[0] #just takes the best one of list of one

print(f"""
number of layers {best_hps.get('num_layers')}.
units per layer {best_hps.get('units')}
learning rate {best_hps.get('learning_rate')}.
batch size {best_hps.get('batch_size')}.
patience {best_hps.get('patience')}.
""")
# Build the model with the optimal hyperparameters and train it on the data for 50 epochs
model = tuner.hypermodel.build(best_hps)
early_stopping = EarlyStopping(monitor='val_loss', patience=best_hps.get('patience'),
                                       restore_best_weights=True)
history = model.fit(X_train, y_train, epochs=500, validation_split=0.3, callbacks=[early_stopping], batch_size=best_hps.get("batch_size"))

val_acc_per_epoch = history.history['val_loss']
print(val_acc_per_epoch)

