import matplotlib.pyplot as plt
import pickle
import numpy as np

"""
Plot your metrics.
"""
fig, axs = plt.subplots(4, 1, figsize=(16, 12))
fig.suptitle(f"Comparison", fontsize=20)

dir_names = ["dagon hyperparams FIFO (0)",
             "dagon hyperparams FIRO (0)",
             "dagon hyperparams RIRO 0.3 (0)",
             "dagon hyperparams GREEDY (0)",
             "dagon hyperparams THRESHOLD 0.3 (0)",
             "dagon hyperparams THRESHOLD_GREEDY 0.3 (0)"]

for dir_name in dir_names:
    with open(f"results/{dir_name}/metrics_results.pkl", 'rb') as file:
            metrics_results = pickle.load(file)

    for i, metric in enumerate(metrics_results.keys()):
        y = metrics_results[metric]
        x = np.arange(0, len(y))
        axs[i].plot(x, y, label=dir_name)
        axs[i].set_title(metric)
        axs[i].legend(loc='upper left', fontsize=10)

plt.tight_layout()
plt.savefig(f"comparison")
plt.show()
plt.close()


