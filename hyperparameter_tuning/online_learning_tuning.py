import tensorflow as tf
from tensorflow import keras
from keras_tuner import BayesianOptimization, HyperModel
from keras.layers import Input, Dense
from keras.models import Model
from keras_uncertainty.models import SimpleEnsemble
from keras.optimizers import Adam
from keras.losses import MeanSquaredError
from src.dataset import SinusiodToyExample, DagonAUVDataset
from keras.callbacks import EarlyStopping

experiment_specification = {
    "EXPERIMENT_IDENTIFIER": "sinus example",
    "BUFFER_SIZE": 100,
    "MODEL_MODE": "FIRO",
    "DATASET_MODE": "subsampled_sequential",
    "DATASET_SIZE": 0.1,
    "BATCH_SIZE": 2,
    "PATIENCE": 10,
    "MAX_EPOCHS": 200,
    "ACCEPT_PROBABILITY": 0.3,
    "INPUT_LAYER_SIZE": 1,
    "OUTPUT_LAYER_SIZE": 1,
    "UNCERTAINTY_THRESHOLD": 0.1
}

dataset = SinusiodToyExample(experiment_specification=experiment_specification)
X_train, y_train = dataset.give_initial_training_set(experiment_specification["BUFFER_SIZE"])


class MyHyperModel(HyperModel):
    def build(self, hp):
        def model_fn():
            # Input layer
            inp = Input(shape=(self.experiment_specification["INPUT_LAYER_SIZE"],))
            hp_units = hp.Int('units', min_value=2, max_value=64, step=1)
            x = Dense(hp_units, activation="relu")(inp)
            # Hyperparam tune number of layers
            num_layers = hp.Int('num_layers', min_value=1, max_value=4, step=1)
            for _ in range(num_layers):
                # for each add units
                
                x = Dense(hp_units, activation="relu")(x)
            mean = Dense(self.experiment_specification["OUTPUT_LAYER_SIZE"], activation="linear")(x)
            train_model = Model(inp, mean)
            hp_learning_rate = hp.Choice('learning_rate', values=[1e-1, 1e-2, 1e-3, 1e-4, 1e-5])
            train_model.compile(loss="mse", optimizer=Adam(learning_rate=hp_learning_rate))
            return train_model

        return SimpleEnsemble(model_fn, num_estimators=10)
    

    def fit(self, hp, model, x, y, validation_data, callbacks=None, **kwargs):
        # Convert the datasets to tf.data.Dataset.
        batch_size = hp.Int("batch_size", 32, 128, step=32, default=64)
        

        # Define the optimizer.
        optimizer = keras.optimizers.Adam(
            hp.Float("learning_rate", 1e-4, 1e-2, sampling="log", default=1e-3)
        )
        loss_fn = keras.losses.SparseCategoricalCrossentropy(from_logits=True)

        # The metric to track validation loss.
        epoch_loss_metric = keras.metrics.Mean()

        # Function to run the train step.
        @tf.function
        def run_train_step(images, labels):
            with tf.GradientTape() as tape:
                logits = model(images)
                loss = loss_fn(labels, logits)
                # Add any regularization losses.
                if model.losses:
                    loss += tf.math.add_n(model.losses)
            gradients = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))

        # Function to run the validation step.
        @tf.function
        def run_val_step(images, labels):
            logits = model(images)
            loss = loss_fn(labels, logits)
            # Update the metric.
            epoch_loss_metric.update_state(loss)

        # Assign the model to the callbacks.
        for callback in callbacks:
            callback.model = model

        # Record the best validation loss value
        best_epoch_loss = float("inf")

        # The custom training loop.
        for epoch in range(2):
            print(f"Epoch: {epoch}")

            # Iterate the training data to run the training step.
            for images, labels in train_ds:
                run_train_step(images, labels)

            # Iterate the validation data to run the validation step.
            for images, labels in validation_data:
                run_val_step(images, labels)

            # Calling the callbacks after epoch.
            epoch_loss = float(epoch_loss_metric.result().numpy())
            for callback in callbacks:
                # The "my_metric" is the objective passed to the tuner.
                callback.on_epoch_end(epoch, logs={"my_metric": epoch_loss})
            epoch_loss_metric.reset_states()

            print(f"Epoch loss: {epoch_loss}")
            best_epoch_loss = min(best_epoch_loss, epoch_loss)

        # Return the evaluation metric value.
        return best_epoch_loss




# https://kegui.medium.com/how-to-tune-the-number-of-epochs-and-batch-size-in-keras-tuner-c2ab2d40878d


# class MyTuner(BayesianOptimization):
#     def run_trial(self, trial, *args, **kwargs):
#         kwargs['batch_size'] = trial.hyperparameters.Choice(
#             'batch_size', values=[1, 2, 4, 8, 16])
#         early_stopping = EarlyStopping(monitor='val_loss',
#                                        patience=trial.hyperparameters.Int(
#                                            'patience', min_value=2, max_value=10),
#                                        restore_best_weights=True
#                                        )
#         if 'callbacks' in kwargs:
#             kwargs['callbacks'].append(early_stopping)
#         else:
#             kwargs['callbacks'] = [early_stopping]
#         return super(MyTuner, self).run_trial(trial, *args, **kwargs)


tuner = BayesianOptimization(hypermodel=MyHyperModel(),
                objective='val_loss',
                directory='hyperparams',
                project_name=experiment_specification["EXPERIMENT_IDENTIFIER"])

# stop_early = EarlyStopping(monitor='val_loss', patience=5)
# , callbacks=[stop_early])
tuner.search(X_train, y_train, epochs=experiment_specification["MAX_EPOCHS"], validation_split=0.3)

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
