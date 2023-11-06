from imports import *
from constants import *

## Plot each iteration
def plot_iteration(dir_name, iteration, y_pred_mean, y_pred_std, current_x_train, current_y_train, picked_x, picked_y, domain, domain_y):
    y_pred_mean = y_pred_mean.reshape((-1,))
    y_pred_std = y_pred_std.reshape((-1,))
    y_pred_up_1 = y_pred_mean + y_pred_std
    y_pred_down_1 = y_pred_mean - y_pred_std
    fig, ax = plt.subplots()
    ax.set_title(f"{STRATEGY} on {MODEL} iteration {iteration}")
    ax.set_ylim([-20.0, 20.0])
    ax.axvline(x=START, color="black", linestyle="dashed")
    ax.axvline(x=END, color="black", linestyle="dashed")
    ax.plot(current_x_train, current_y_train, '.', color=(
        0.9, 0, 0, 0.5), markersize=15, label="training set")
    if len(picked_y) != 0:
        ax.plot(picked_x, picked_y, '.', color=(1, 1, 0.1, 1),
            markersize=25, label="new points")
    ax.plot(domain, domain_y, '.', color=(0, 0.9, 0, 1),
            markersize=3, label="ground truth")
    ax.fill_between(domain.ravel(), y_pred_down_1,
                    y_pred_up_1,  color=(0, 0.5, 0.9, 0.5))
    ax.plot(domain.ravel(), y_pred_mean, '.',
            color=(1, 1, 1, 0.8), markersize=0.2)
    ax.legend(bbox_to_anchor=(1, 1))#, loc='upper left')
    # save plot
    # plt.show()
    plt.savefig(f"{dir_name}/iteration {iteration}")
    plt.close()

def plot_metrics(dir_name, maes, r2s):
    # Draw stuff
    # Create a figure and three subplots arranged vertically
    x = np.arange(len(maes))
    fig, (ax1, ax3) = plt.subplots(2, 1, figsize=(16, 12))
    fig.suptitle('Online Learning Metrics', fontsize=16)
    # Plot data on the first subplot
    ax1.plot(x, maes, label='MAE')
    ax1.set_title('MAE')
    ax1.legend()
    ax3.plot(x, r2s, label='R2')
    ax3.set_title('Coefficient of Determination')
    ax3.legend()

    # Show the plots
    plt.tight_layout()
    plt.savefig(f"{dir_name}/metrics")
    # plt.show()
    plt.close()