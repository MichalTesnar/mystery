import matplotlib.pyplot as plt
import pickle

prefix = "Full data "
plot_name = "Analyse Uncertainty"
fig, axs = plt.subplots(1, 1, figsize=(16, 13))
fig.suptitle(f"{plot_name}", fontsize=15)

dir_name = "FIFO tuned (0)"
with open(f"results/{prefix}{dir_name}/metrics_results.pkl", 'rb') as file:
    metrics_results_FIFO = pickle.load(file)
dir_name = "FIRO tuned (0)"
with open(f"results/{prefix}{dir_name}/metrics_results.pkl", 'rb') as file:
    metrics_results_FIRO = pickle.load(file)

uncertainties_FIFO = metrics_results_FIFO["Prediction Uncertainty"]
uncertainties_FIRO = metrics_results_FIRO["Prediction Uncertainty"]
uncertainties_FIFO.sort() # small to big
uncertainties_FIRO.sort() # small to big

length = len(uncertainties_FIFO)
for i in range(1, 10):
    mid = (uncertainties_FIFO[int(length*i/10)] + uncertainties_FIRO[int(length*i/10)])/2
    print(f"{mid:.4f}")