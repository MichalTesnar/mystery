"""
IDEA:
Number of data points in the buffer -- set to a fixed number, let us say 10% of the dataset?
Take the first 10% of the data, cause that is all that you can get, cross-validate on it
With fixed architecture can be searched once, then kept fixed for the rest of the time.
"""

"""
How to find the parameters of the model: complexity analysis is difficult,
we could be looking at the equations, but I feel like cross-validation
is more robust and fewer assumptions.
"""

"""
HYPERPARAMETERS TO SEARCH
  - MLP size: layers, neurons
  - training: batch_size, learning rate, patience
DECIDE TO KEEP FIXED
  - MSE loss
  - Optimizer Adam
  - Activation: relu->relu + linear at the end
"""

"""
Questions
Hyperparameters
- implement l2? Any form of regularization?

UQ Hyperparams
- number of estimators: better uncertainty, slower approximation?
- threshold of the methods?
- 
"""


import tensorflow as tf
from tensorflow import keras
import keras_tuner as kt
from src.dataset import SinusiodToyExample, DagonAUVDataset
experiment_specification = {
    "EXPERIMENT_IDENTIFIER": "FIRO genuine test",
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
X_train, y_train = dataset.give_initial_training_set(
    experiment_specification["BUFFER_SIZE"])


def model_builder(hp):
    model = keras.Sequential()
    model.add(keras.layers.Input(shape=(1)))

    num_layers = hp.Int('num_layers', min_value=1,
                        max_value=10, default=3, step=1)

    for i in range(num_layers):
        hp_units = hp.Int('units', min_value=2, max_value=8, step=2)
        model.add(keras.layers.Dense(units=hp_units, activation='relu'))

    model.add(keras.layers.Dense(1, activation="linear"))
    hp_learning_rate = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])

    model.compile(optimizer=keras.optimizers.Adam(learning_rate=hp_learning_rate),
                  loss=keras.losses.MeanSquaredError())

    return model


class MyTuner(kt.tuners.BayesianOptimization):
    def run_trial(self, trial, *args, **kwargs):
        kwargs['batch_size'] = trial.hyperparameters.Int(
            'batch_size', 1, 8, step=1)
        early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss',
            patience=trial.hyperparameters.Int('patience', 3, 10),
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
                project_name='trial_patience')

stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)
tuner.search(X_train, y_train, epochs=1000,
             validation_split=0.2, callbacks=[stop_early])

# Get the optimal hyperparameters
best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]

print(f"""
{best_hps.get('units')}
{best_hps.get('learning_rate')}.
{best_hps.get('num_layers')}.
{best_hps.get('batch_size')}.
{best_hps.get('patience')}.
""")
exit()

# Build the model with the optimal hyperparameters and train it on the data for 50 epochs
model = tuner.hypermodel.build(best_hps)
history = model.fit(X_train, y_train, epochs=50, validation_split=0.2)

val_acc_per_epoch = history.history['val_loss']
best_epoch = val_acc_per_epoch.index(max(val_acc_per_epoch)) + 1
print('Best epoch: %d' % (best_epoch,))

print(f"""
{best_hps.get('units')}
{best_hps.get('learning_rate')}.
{best_hps.get('num_layers')}.
""")
