import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Read the data
csv_file_path = "../dagon_dataset.csv"
df = pd.read_csv(csv_file_path, skiprows=1, nrows=2000)
u = df.iloc[:, 0]
v = df.iloc[:, 1]
r = df.iloc[:, 2]
th1 = df.iloc[:, 3]
th2 = df.iloc[:, 4]
th3 = df.iloc[:, 5]
x = np.arange(0, len(th1))

# Create a figure and three subplots arranged vertically
fig, (ax1, ax2, ax3, ax4, ax5, ax6) = plt.subplots(6, 1, figsize=(8, 12), sharex=True)
fig.suptitle('Input Features of AUV Dagon Dataset', fontsize=20)

# Plot data
ax1.plot(x, u, '.', markersize=1, label=r'$u$ $(m/s)$')
ax2.plot(x, v, '.', markersize=1, label=r'$v$ $(m/s)$')
ax3.plot(x, r, '.', markersize=1, label=r'$r$ $(rd/s)$')
ax4.plot(x, th1, '.', markersize=1, label=r'$n_1$ $(rps)$')
ax5.plot(x, th2, '.', markersize=1, label=r'$n_2$ $(rps)$')
ax6.plot(x, th3, '.', markersize=1, label=r'$n_3$ $(rps)$')
ax1.legend(loc='upper left', fontsize="20")
ax2.legend(loc='upper left', fontsize="20")
ax3.legend(loc='upper left', fontsize="20")
ax4.legend(loc='upper left', fontsize="20")
ax5.legend(loc='upper left', fontsize="20")
ax6.legend(loc='upper left', fontsize="20")

# Show the plots
plt.tight_layout()
plt.savefig("input_features.png")
plt.show()
