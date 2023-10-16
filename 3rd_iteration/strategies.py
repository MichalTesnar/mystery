from utils import *
from constants import *
from imports import *
from utils import *
from plotting import *

# Update training data
def apply_strategy(iteration, x_train, y_train, current_x_train, current_y_train, current_model, data_index, dir_name="", maes=[], errors=[]):
        if STRATEGY == "DROP_LAST":
            current_x, current_y = x_train[NEW_DATA_RATE + NEW_PER_ITER*iteration: NEW_DATA_RATE + NEW_PER_ITER*(
            iteration+1)], y_train[NEW_DATA_RATE + NEW_PER_ITER*iteration:NEW_DATA_RATE + NEW_PER_ITER*(iteration+1)]
            # replace the oldest of the previous with it
            current_x_train = np.concatenate(
                (current_x_train[NEW_PER_ITER:], current_x))
            current_y_train = np.concatenate(
                (current_y_train[NEW_PER_ITER:], current_y))
            picked_x = current_x
            picked_y = current_y

        if STRATEGY == "DROP_RANDOM":
            current_x, current_y = x_train[NEW_DATA_RATE + NEW_PER_ITER*iteration: NEW_DATA_RATE + NEW_PER_ITER*(
            iteration+1)], y_train[NEW_DATA_RATE + NEW_PER_ITER*iteration:NEW_DATA_RATE + NEW_PER_ITER*(iteration+1)]
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
                # if MODEL == "GRP":
                #     predicted_std = [predicted_std]
                # print(predicted_std)
                while np.min(predicted_stds) > predicted_std[0][0] or predicted_std[0][0] < THRESHOLD:
                    k += 1
                    data_index += 1
                    if data_index + NEW_DATA_RATE + 1 == SAMPLE_RATE:
                        print("Run out of data")
                        if len(maes) == 0:
                             x_train, y_train, domain, domain_y = get_data()
                             y_pred_mean, y_pred_std = pred_model(MODEL, current_model, domain)
                             plot_iteration(dir_name, iteration, y_pred_mean, y_pred_std, current_x_train, current_y_train, picked_x, picked_y, domain, domain_y)
                             exit()
                        last_mae = maes[-1]
                        last_error = errors[-1]
                        lenght_to_pad = ITERATIONS - len(maes)
                        maes = np.pad(maes, (0, lenght_to_pad), constant_values=last_mae)
                        errors = np.pad(errors, (0, lenght_to_pad), constant_values=last_error)
                        
                        if PLOT_METRICS:
                            plot_metrics(dir_name, maes, errors)
                        exit()
                    current_x = x_train[NEW_DATA_RATE + data_index: NEW_DATA_RATE + data_index + 1]
                    current_y = y_train[NEW_DATA_RATE + data_index: NEW_DATA_RATE + data_index + 1]
                    _, predicted_std = pred_model(MODEL, current_model, current_x.reshape((-1, 1)))

                data_index += 1
                if data_index + NEW_DATA_RATE + 1 == SAMPLE_RATE:
                        print("Run out of data")
                        last_mae = maes[-1]
                        last_error = errors[-1]
                        lenght_to_pad = ITERATIONS - len(maes)
                        maes = np.pad(maes, (0, lenght_to_pad), constant_values=last_mae)
                        errors = np.pad(errors, (0, lenght_to_pad), constant_values=last_error)
                        if PLOT_METRICS:
                            plot_metrics(dir_name, maes, errors)
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
                push_x = np.tile(current_x[j], min(NEW_DATA_RATE, max(iteration, 1)))
                push_y = np.tile(current_y[j], min(NEW_DATA_RATE, max(iteration, 1)))
                current_model = retrain_model(MODEL, current_model, push_x, push_y)

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
        return current_model, current_x_train, current_y_train, picked_x, picked_y, data_index