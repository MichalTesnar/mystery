import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Specify the path to your CSV file
csv_file_path = "dagon_dataset.csv"
df = pd.read_csv(csv_file_path, skiprows=1)
df = df.head(1000)

# Obtain columns from the dataframe
u = df.iloc[:, 0]
v = df.iloc[:, 1]
r = df.iloc[:, 2]
du = df.iloc[:, 6]
dv = df.iloc[:, 7]
dr = df.iloc[:, 8]

# Create a 3D figure
fig = plt.figure(figsize=(16, 12))
ax = fig.add_subplot(111, projection='3d')
ax.set_title("Trajectory with Differentials of AUV Dagon Captured in the Dataset")

# Colors for the scatter points
cmap = plt.get_cmap('viridis')
colors = np.arange(len(u))
normalize = plt.Normalize(colors.min(), colors.max())
colormap = cmap(normalize(colors))

# Scatter plot the points
ax.scatter(u, v, r, c=colormap, marker='o')
ax.set_xlabel('u')
ax.set_ylabel('v')
ax.set_zlabel('r')

# Create arrows in a loop
for x0, y0, z0, u, v, w in zip(u, v, r, du, dv, dr):
    ax.quiver(x0, y0, z0, u, v, w, color='r')

# Show the plot
plt.show()
