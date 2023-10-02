"""
"""

from imports import *
from utils import *
from constants import *

# fixing ensembles
from tensorflow.python.framework.ops import disable_eager_execution
disable_eager_execution()


#### Possible models
MODEL = "Ensembles"
# MODEL = "Dropout"

#### Possible strategies
# STRATEGY = "DROP_LAST"
# STRATEGY = "DROP_RANDOM"
STRATEGY = "ACTIVE_BUFFER"
# STRATEGY = "ACTIVE_BUFFER_BOOSTED"
# STRATEGY = "HEURISTIC_CLOSEST"

if __name__ == "__main__":
    # make dir for plot
    dir_i = 0
    while os.path.isdir(f"figs/{STRATEGY} on {MODEL} ({dir_i})"):
        dir_i += 1
    dir = f"figs/{STRATEGY} on {MODEL} ({dir_i})"
    os.mkdir(f"figs/{STRATEGY} on {MODEL} ({dir_i})")

    # data
    x_train = np.linspace(-4.0, 4.0, num=SAMPLE_RATE)
    train_indices = np.arange(SAMPLE_RATE)
    np.random.shuffle(train_indices)
    x_train = x_train[train_indices]
    y_train = toy_function(x_train)
    domain = np.linspace(-4.0, 4.0, num=100)
    domain = domain.reshape((-1, 1))
    domain_y = toy_function(domain)
    # define model to be retrained later
    current_model = get_model(MODEL)
    current_x_train, current_y_train = x_train[0:NEW_DATA_RATE], y_train[0:NEW_DATA_RATE] # grab empty at the start
    
    print(f"The files will be saved in {dir}")
    # iteratively retrain on new data
    for i in range(ITERATIONS):
        # grab new points
        current_x, current_y = x_train[NEW_DATA_RATE + NEW_PER_ITER*i: NEW_DATA_RATE + NEW_PER_ITER*(i+1)], y_train[NEW_DATA_RATE + NEW_PER_ITER*i:NEW_DATA_RATE + NEW_PER_ITER*(i+1)]
        # Update training data
        if STRATEGY == "DROP_LAST":
            # replace the oldest of the previous with it
            current_x_train = np.concatenate((current_x_train[NEW_PER_ITER:], current_x))
            current_y_train = np.concatenate((current_y_train[NEW_PER_ITER:], current_y))
            picked_x = current_x
            picked_y = current_y
            
        if STRATEGY == "DROP_RANDOM":
            # shuffle indices of the previous data, take the appropriate subset of training and append the new data
            remaining_indices = np.arange(NEW_DATA_RATE - NEW_PER_ITER)
            np.random.shuffle(remaining_indices)
            current_x_train = np.concatenate((current_x_train[remaining_indices], current_x))
            current_y_train = np.concatenate((current_y_train[remaining_indices], current_y))
            picked_x = current_x
            picked_y = current_y

        if STRATEGY == "ACTIVE_BUFFER":
            picked_x = []
            picked_y = []
            _, predicted_stds = pred_model(MODEL, current_model, current_x_train.reshape((-1, 1)))
            _, predicted_std = pred_model(MODEL, current_model, current_x.reshape((-1, 1)))

            for j, pre in enumerate(predicted_std):
                if min(predicted_stds) < pre:
                    idx = np.argmin(predicted_stds)
                    predicted_stds[idx] = pre
                    current_x_train[idx] = current_x[j]
                    picked_x.append(current_x[j])
                    current_y_train[idx] = current_y[j]
                    picked_y.append(current_y[j])

        if STRATEGY == "ACTIVE_BUFFER_BOOSTED":
            picked_x = []
            picked_y = []
            _, stds_old_data = pred_model(MODEL, current_model, current_x_train.reshape((-1, 1)))
            _, stds_new_data = pred_model(MODEL, current_model, current_x.reshape((-1, 1)))

            for j, pred in enumerate(stds_new_data): # iterate over the newly evaluated data
                if min(stds_old_data) < pred: # if current point has higher uncertainty than the minimum in the current datase
                    idx = np.argmin(stds_old_data) # grab the index of the minimum in the old dataset
                    stds_old_data[idx] = pred # replace this item with the new uncertainty
                    # store the appropriate data
                    current_x_train[idx] = current_x[j]
                    picked_x.append(current_x[j])
                    current_y_train[idx] = current_y[j]
                    picked_y.append(current_y[j])
                # give the model extra push -- train at most new_data_rate steps
                push_x = np.tile(current_x[j], min(NEW_DATA_RATE, max(i, 1)))
                push_y = np.tile(current_y[j], min(NEW_DATA_RATE, max(i, 1)))
                current_model = retrain_model(MODEL, current_model, push_x, push_y)

        if STRATEGY == "HEURISTIC_CLOSEST":
            picked_x = []
            picked_y = []
            # one by one check
            for x, y in zip(current_x, current_y):
                # calculate the distances
                distances = np.array([[abs(x - y) for x in current_x_train] for y in current_x_train]) + 10**6 * np.eye(current_x_train.shape[0])
                # print(distances)
                distances_of_new_x = np.abs(current_x_train - x)
                # print(distances_of_new_x)
                if np.min(distances_of_new_x) > np.min(distances): # if our new point is more far away then the other points are far away from each other
                    cord = np.argmin(distances) # point that is closest to the all other points, we want to remove it
                    x_cord, y_cord = cord // distances.shape[0], cord % distances.shape[0]
                    # replace the point in the dataset
                    current_x_train[x_cord] = x 
                    current_y_train[x_cord] = y 
                    # record this
                    picked_x.append(x)
                    picked_y.append(y)

        # Retrain & predict
        current_model = retrain_model(MODEL, current_model, current_x_train, current_y_train)
        y_pred_mean, y_pred_std = pred_model(MODEL, current_model, domain)
        # Metrics
        score = gaussian_interval_score(domain_y, y_pred_mean, y_pred_std)
        calib_err = regressor_calibration_error(y_pred_mean, domain_y, y_pred_std)
        print(f"iteration: {i} score: {score:.2f} calib_err: {calib_err:.2f}")
        
        # Aggregate the prediction
        y_pred_mean = y_pred_mean.reshape((-1,))
        y_pred_std = y_pred_std.reshape((-1,))
        y_pred_up_1 = y_pred_mean + y_pred_std
        y_pred_down_1 = y_pred_mean - y_pred_std
        # plot data
        if PLOT:
            # Draw stuff
            fig, ax = plt.subplots(nrows=1, ncols=len(["Dropout"]), figsize=(20, 3))
            ax.set_title(f"{STRATEGY} on {MODEL} iteration {i}")
            ax.set_ylim([-20.0, 20.0])
            ax.axvline(x=-4.0, color="black", linestyle="dashed")
            ax.axvline(x= 4.0, color="black", linestyle="dashed")
            ax.get_xaxis().set_ticks([])
            ax.get_yaxis().set_ticks([])    
            ax.plot(current_x_train, current_y_train, '.', color=(0.9, 0, 0, 0.5), markersize=15, label="training set")
            ax.plot(picked_x, picked_y, '.', color=(1, 1, 0.1, 0.5), markersize=15, label="new points")
            ax.plot(domain, domain_y, '.', color=(0, 0.9, 0, 1), markersize=3, label="ground truth")
            ax.fill_between(domain.ravel(), y_pred_down_1, y_pred_up_1,  color=(0, 0.5, 0.9, 0.5))
            ax.plot(domain.ravel(), y_pred_mean, '.', color=(1, 1, 1, 0.8), markersize=0.2)
            ax.legend(bbox_to_anchor=(1, 1), loc='upper left')
            # save plot
            plt.savefig(f"{dir}/iteration {i}")
                