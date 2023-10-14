"""
"""

from imports import *
from utils import *
from constants import *

# fixing ensembles
from tensorflow.python.framework.ops import disable_eager_execution
disable_eager_execution()

if __name__ == "__main__":
    # make dir for plot
    if PLOT:
        dir_i = 0
        while os.path.isdir(f"figs/{STRATEGY} on {MODEL} ({dir_i})"):
            dir_i += 1
        dir_name = f"figs/{STRATEGY} on {MODEL} ({dir_i})"
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
    # grab empty at the start
    current_x_train, current_y_train = x_train[0:NEW_DATA_RATE], y_train[0:NEW_DATA_RATE]
    # extra pre-fitting
    current_model = retrain_model(
        MODEL, current_model, current_x_train, current_y_train, extra_epochs=EXTRA_EPOCHS)
    print(f"The files will be saved in {dir_name}")
    # collect your metric
    maes = []
    errors = [] 
    data_index = 1
    # iteratively retrain on new data
    for i in tqdm(range(1, ITERATIONS+1)):
        # grab new points
        
        
        # Update training data
        if STRATEGY == "DROP_LAST":
            current_x, current_y = x_train[NEW_DATA_RATE + NEW_PER_ITER*i: NEW_DATA_RATE + NEW_PER_ITER*(
            i+1)], y_train[NEW_DATA_RATE + NEW_PER_ITER*i:NEW_DATA_RATE + NEW_PER_ITER*(i+1)]
            # replace the oldest of the previous with it
            current_x_train = np.concatenate(
                (current_x_train[NEW_PER_ITER:], current_x))
            current_y_train = np.concatenate(
                (current_y_train[NEW_PER_ITER:], current_y))
            picked_x = current_x
            picked_y = current_y

        if STRATEGY == "DROP_RANDOM":
            current_x, current_y = x_train[NEW_DATA_RATE + NEW_PER_ITER*i: NEW_DATA_RATE + NEW_PER_ITER*(
            i+1)], y_train[NEW_DATA_RATE + NEW_PER_ITER*i:NEW_DATA_RATE + NEW_PER_ITER*(i+1)]
            # shuffle indices of the previous data, take the appropriate subset of training and append the new data
            remaining_indices = np.random.choice(np.arange(NEW_DATA_RATE), NEW_DATA_RATE - NEW_PER_ITER, replace=False) 
            current_x_train = np.concatenate(
                (current_x_train[remaining_indices], current_x))
            current_y_train = np.concatenate(
                (current_y_train[remaining_indices], current_y))
            picked_x = current_x
            picked_y = current_y

        if STRATEGY == "ACTIVE_BUFFER":
            picked_x = []
            picked_y = []
            cnt = 0
            for _ in range(NEW_PER_ITER):
                
                current_x = x_train[NEW_DATA_RATE + data_index: NEW_DATA_RATE + data_index + 1]
                current_y = y_train[NEW_DATA_RATE + data_index: NEW_DATA_RATE + data_index + 1]
                _, predicted_stds = pred_model(MODEL, current_model, current_x_train.reshape((-1, 1)))
                _, predicted_std = pred_model(MODEL, current_model, current_x.reshape((-1, 1)))
                k = 0
                
                while np.min(predicted_stds) > predicted_std or predicted_std[0][0] < THRESHOLD:
                    k += 1
                    data_index += 1
                    if data_index + NEW_DATA_RATE + 1 == SAMPLE_RATE:
                        print("Run out of data")
                        exit()
                    current_x = x_train[NEW_DATA_RATE + data_index: NEW_DATA_RATE + data_index + 1]
                    current_y = y_train[NEW_DATA_RATE + data_index: NEW_DATA_RATE + data_index + 1]
                    _, predicted_std = pred_model(MODEL, current_model, current_x.reshape((-1, 1)))

                data_index += 1
                if data_index + NEW_DATA_RATE + 1 == SAMPLE_RATE:
                        print("Run out of data")
                        exit()
                cnt += k
                idx = np.argmin(predicted_stds)
                predicted_stds[idx] = predicted_std
                current_x_train[idx] = current_x[0]
                picked_x.append(current_x[0])
                current_y_train[idx] = current_y[0]
                picked_y.append(current_y[0])
            
            print("Skipped:", cnt)
                    

        if STRATEGY == "ACTIVE_BUFFER_BOOSTED":
            picked_x = []
            picked_y = []
            _, stds_old_data = pred_model(
                MODEL, current_model, current_x_train.reshape((-1, 1)))
            _, stds_new_data = pred_model(
                MODEL, current_model, current_x.reshape((-1, 1)))

            # iterate over the newly evaluated data
            for j, pred in enumerate(stds_new_data):
                # if current point has higher uncertainty than the minimum in the current datase
                if min(stds_old_data) < pred:
                    # grab the index of the minimum in the old dataset
                    idx = np.argmin(stds_old_data)
                    # replace this item with the new uncertainty
                    stds_old_data[idx] = pred
                    # store the appropriate data
                    current_x_train[idx] = current_x[j]
                    picked_x.append(current_x[j])
                    current_y_train[idx] = current_y[j]
                    picked_y.append(current_y[j])
                # give the model extra push -- train at most new_data_rate steps
                push_x = np.tile(current_x[j], min(NEW_DATA_RATE, max(i, 1)))
                push_y = np.tile(current_y[j], min(NEW_DATA_RATE, max(i, 1)))
                current_model = retrain_model(
                    MODEL, current_model, push_x, push_y)

        if STRATEGY == "HEURISTIC_CLOSEST":
            picked_x = []
            picked_y = []
            # one by one check
            for x, y in zip(current_x, current_y):
                # calculate the distances
                distances = np.array([[abs(x - y) for x in current_x_train]
                                     for y in current_x_train]) + 10**6 * np.eye(current_x_train.shape[0])
                distances_of_new_x = np.abs(current_x_train - x)
                # if our new point is more far away then the other points are far away from each other
                if np.min(distances_of_new_x) > np.min(distances):
                    # point that is closest to the all other points, we want to remove it
                    cord = np.argmin(distances)
                    x_cord, y_cord = cord // distances.shape[0], cord % distances.shape[0]
                    # replace the point in the dataset
                    current_x_train[x_cord] = x
                    current_y_train[x_cord] = y
                    # record this
                    picked_x.append(x)
                    picked_y.append(y)

        # Retrain & predict
        current_model = retrain_model(MODEL, current_model, current_x_train, current_y_train, batch_size=2)
        # Collecting metrics
        pred_mean, pred_std = pred_model(MODEL, current_model, current_x_train.reshape(-1,1))
        print(pred_std)
        pred_mean, pred_std = pred_model(MODEL, current_model, domain)
        difs = domain_y.reshape(-1,1)-pred_mean
        mae = np.sum(abs(difs))/SAMPLE_RATE
        maes.append(mae)
        #  Aggregate the prediction
        y_pred_mean = pred_mean
        y_pred_std = pred_std
        y_pred_mean = y_pred_mean.reshape((-1,))
        y_pred_std = y_pred_std.reshape((-1,))
        y_pred_up_1 = y_pred_mean + y_pred_std
        y_pred_down_1 = y_pred_mean - y_pred_std
        calib_err = regressor_calibration_error(y_pred_mean, domain_y, y_pred_std)
        errors.append(calib_err)

        if PLOT:
            # Draw stuff
            fig, ax = plt.subplots(nrows=1, ncols=len(
                ["Dropout"]), figsize=(20, 3))
            ax.set_title(f"{STRATEGY} on {MODEL} iteration {i}")
            ax.set_ylim([-20.0, 20.0])
            ax.axvline(x=-4.0, color="black", linestyle="dashed")
            ax.axvline(x=4.0, color="black", linestyle="dashed")
            ax.get_xaxis().set_ticks([])
            ax.get_yaxis().set_ticks([])
            ax.plot(current_x_train, current_y_train, '.', color=(
                0.9, 0, 0, 0.5), markersize=15, label="training set")
            ax.plot(picked_x, picked_y, '.', color=(1, 1, 0.1, 1),
                    markersize=25, label="new points")
            ax.plot(domain, domain_y, '.', color=(0, 0.9, 0, 1),
                    markersize=3, label="ground truth")
            ax.fill_between(domain.ravel(), y_pred_down_1,
                            y_pred_up_1,  color=(0, 0.5, 0.9, 0.5))
            ax.plot(domain.ravel(), y_pred_mean, '.',
                    color=(1, 1, 1, 0.8), markersize=0.2)
            ax.legend(bbox_to_anchor=(1, 1), loc='upper left')
            # save plot
            # plt.show()
            plt.savefig(f"{dir_name}/iteration {i}")
            plt.close()
    if PLOT:
        # Draw stuff
        # Create a figure and three subplots arranged vertically
        x = np.arange(ITERATIONS)
        fig, (ax1, ax3) = plt.subplots(2, 1, figsize=(16, 12))
        fig.suptitle('Online Learning Metrics', fontsize=16)
        # Plot data on the first subplot
        ax1.plot(x, maes, label='MAE')
        ax1.set_title('MAE')
        ax1.legend()
        ax3.plot(x, errors, label='Calibration Errors')
        ax3.set_title('Calibration Errors')
        ax3.legend()

        # Show the plots
        plt.tight_layout()
        plt.savefig(f"{dir_name}/metrics")
        # plt.show()
        plt.close()
        
