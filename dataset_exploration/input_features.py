import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Read the data
csv_file_path = "dagon_dataset.csv"
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
fig.suptitle('AUV Dagon Dataset Input Features', fontsize=20)

# Plot data
ax1.plot(x, u, '.', markersize=1, label=r'$u$ $(m/s)$')
ax2.plot(x, v, '.', markersize=1, label=r'$v$ $(m/s)$')
ax3.plot(x, r, '.', markersize=1, label=r'$r$ $(rd/s)$')
ax4.plot(x, th1, '.', markersize=1, label=r'$n_1$ $(rps)$')
ax5.plot(x, th2, '.', markersize=1, label=r'$n_2$ $(rps)$')
ax6.plot(x, th3, '.', markersize=1, label=r'$n_3$ $(rps)$')

FONTSIZE = 15
ax6.set_xlabel(r'$t(s)$', fontsize=FONTSIZE)

ax1.set_ylabel(r'$u$ $(m/s)$', fontsize=FONTSIZE)
ax2.set_ylabel(r'$v$ $(m/s)$', fontsize=FONTSIZE)
ax3.set_ylabel(r'$r$ $(rd/s)$', fontsize=FONTSIZE)
ax4.set_ylabel(r'$n_1$ $(rps)$', fontsize=FONTSIZE)
ax5.set_ylabel(r'$n_2$ $(rps)$', fontsize=FONTSIZE)
ax6.set_ylabel(r'$n_3$ $(rps)$', fontsize=FONTSIZE)

# Show the plots
plt.tight_layout()
plt.savefig("input_features.pdf")
plt.show()
