from extract import Dataset
from keras.models import Sequential
from keras.layers import Dense
import numpy as np
import matplotlib.pyplot as plt

# Dataset
datasetSQ = Dataset(mode="subsumpled_sequential")
X_train, y_train = datasetSQ.get_training_set()
X_val, y_val = datasetSQ.get_validation_set()
X_test, y_test = datasetSQ.get_test_set()

# Model definitions
model = Sequential()
model.add(Dense(64, activation="relu", input_shape=(6,)))
model.add(Dense(64, activation="relu"))
model.add(Dense(32, activation="relu"))
model.add(Dense(3, activation="linear"))
model.compile(loss="mean_squared_error", optimizer="adam")

# Training
EPOCHS = 5
history = model.fit(X_train, y_train, epochs=EPOCHS, validation_data=(X_val, y_val), batch_size=2)

# Plotting
x = np.arange(EPOCHS)
plt.plot(x, history.history['loss'])
plt.plot(x, history.history['val_loss'])
plt.title("Some like graph or something")
plt.show()