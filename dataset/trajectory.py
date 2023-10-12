"""
Just a quick note, these are actually not trajectories, these are velocities, and accelerations are pictured as velocity.
Happens. LOL.
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Specify the path to your CSV file
csv_file_path = "dagon_dataset.csv"
df = pd.read_csv(csv_file_path, skiprows=1)
u = df.iloc[:, 0]
v = df.iloc[:, 1]
r = df.iloc[:, 2]

# Create a 3D figure
fig = plt.figure(figsize=(16, 12))
ax = fig.add_subplot(111, projection='3d')

# Colors
cmap = plt.get_cmap('viridis')
colors = np.arange(len(u))
normalize = plt.Normalize(colors.min(), colors.max())
colormap = cmap(normalize(colors))

# Scatter plot the points
ax.scatter(u, v, r, c=colormap, marker='o')
ax.set_title("Trajectory of AUV Dagon Captured in the Dataset")
ax.set_xlabel('u')
ax.set_ylabel('v')
ax.set_zlabel('r')
plt.show()