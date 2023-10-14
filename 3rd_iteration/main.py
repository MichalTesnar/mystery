"""
"""

from imports import *
from utils import *
from constants import *
from plotting import *
from strategies import *

# fixing ensembles
from tensorflow.python.framework.ops import disable_eager_execution
disable_eager_execution()

if __name__ == "__main__":
    # make dir for plot
    if PLOT_EACH_ITERATION or PLOT_TRAINING or PLOT_METRICS:
        dir_i = 0
        while os.path.isdir(f"figs/{STRATEGY} on {MODEL} ({dir_i})"):
            dir_i += 1
        dir_name = f"figs/{STRATEGY} on {MODEL} ({dir_i})"
        os.mkdir(f"figs/{STRATEGY} on {MODEL} ({dir_i})")
        print(f"The files will be saved in {dir_name}")

    # data
    x_train, y_train, domain, domain_y = get_data()
    current_x_train, current_y_train = x_train[0:NEW_DATA_RATE], y_train[0:NEW_DATA_RATE] # grab empty at the start

    # define model and train
    current_model = get_model(MODEL)
    current_model = retrain_model(MODEL, current_model, current_x_train, current_y_train, extra_epochs=EXTRA_EPOCHS) # extra pre-fitting

    # collect your metric
    maes = []
    errors = [] 
    data_index = 1
    
    # iteratively retrain on new data
    for i in tqdm(range(1, ITERATIONS+1)):
        # obtain training data based on strategy
        current_model, current_x_train, current_y_train, picked_x, picked_y, data_index = apply_strategy(i, x_train, y_train, current_x_train, current_y_train, current_model, data_index)
        # Retrain
        current_model = retrain_model(MODEL, current_model, current_x_train, current_y_train, batch_size=2)
        # Predict & Collect Metrics
        pred_mean, pred_std = pred_model(MODEL, current_model, domain)
        difs = domain_y.reshape(-1,1)-pred_mean
        mae = np.sum(abs(difs))/SAMPLE_RATE
        maes.append(mae)
        calib_err = regressor_calibration_error(pred_mean.reshape(-1,), domain_y, pred_std.reshape(-1,))
        errors.append(calib_err)

        if PLOT_EACH_ITERATION:
            plot_iteration(dir_name, i, pred_mean, pred_std, current_x_train, current_y_train, picked_x, picked_y, domain, domain_y)
        if PLOT_TRAINING and MODEL != "GRP": # for some models we can extract the training history
            plot_iteration(dir_name, i, pred_mean, pred_std, current_x_train, current_y_train, picked_x, picked_y, domain, domain_y)
    if PLOT_METRICS:
        plot_metrics(dir_name, maes, errors)
        
