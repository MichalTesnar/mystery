import numpy as np
import math
import os

from keras.models import Sequential, Model
from keras.layers import Dense, Input

from keras_uncertainty.models import StochasticRegressor, DeepEnsembleRegressor
from keras_uncertainty.layers import StochasticDropout
from keras_uncertainty.metrics import gaussian_interval_score
from keras_uncertainty.utils import regressor_calibration_error
from keras_uncertainty.losses import regression_gaussian_nll_loss

import matplotlib.pyplot as plt