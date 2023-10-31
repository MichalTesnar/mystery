# Constants
NUM_SAMPLES = 20  # number of samples for the network when it runs estimation
# number of epochs to (re)fit the model on the newly observed data
EPOCHS = 1000000
SAMPLE_RATE = 400  # the rate at which we sample the interval we want to train on
NEW_DATA_RATE = 100 # size of the buffer for the methods
NEW_PER_ITER = 1 # how much data we add each time
ITERATIONS = int((SAMPLE_RATE - NEW_DATA_RATE)/NEW_PER_ITER)  # iterations to be plotted
EXTRA_EPOCHS = 0
THRESHOLD = 0.35
PATIENCE = 10

#### PLOTTING
PLOT_EACH_ITERATION = True
PLOT_TRAINING = False
PLOT_METRICS = True
START = -6.0
END = 6.0

# Possible models
MODEL = "Ensembles"
# MODEL = "Dropout"
# MODEL = "GRP"

# Possible strategies
# STRATEGY = "ACTIVE_BUFFER_BOOSTED"
STRATEGY = "ACTIVE_BUFFER"  
# STRATEGY = "DROP_LAST"
# STRATEGY = "DROP_RANDOM"
# STRATEGY = "HEURISTIC_CLOSEST"