import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from matplotlib import cm
from matplotlib import colors

# Read final_losses.csv
filename = 'final_losses_lr0.8_epochs5000_or.csv'
df = pd.read_csv(filename)

# from filename, parse everything after final_losses_ and before .csv
params = filename.split('final_losses_')[1].split('.csv')[0]

# Rename columns
df.rename(columns={'Final Loss': 'final_loss', 'Activation': 'activation_function', 'Loss': 'loss_function'}, inplace=True)

# Create mappings for activation_function and loss_function
activation_mapping = {name: idx for idx, name in enumerate(df['activation_function'].unique())}
loss_mapping = {name: idx for idx, name in enumerate(df['loss_function'].unique())}

# Replace strings with numeric values
df['activation_function'] = df['activation_function'].map(activation_mapping)
df['loss_function'] = df['loss_function'].map(loss_mapping)

# Create a colormap
cmap = cm.get_cmap('viridis', 10)  # 10 colors
norm = colors.Normalize(vmin=0, vmax=10)

# Create a 3D plot
fig = plt.figure(figsize=(15, 10))
ax = fig.add_subplot(111, projection='3d')

# Create a scatter plot


df['final_loss'] = np.log10(df['final_loss'])
ax.scatter(df['activation_function'], df['loss_function'], df['final_loss'], c=df['final_loss'], cmap=cmap, norm=norm)

#X, Y = np.meshgrid(df['activation_function'], df['loss_function'])
#X, Y = np.meshgrid(df['activation_function'].unique(), df['loss_function'].unique())
#Z = df.pivot(index='loss_function', columns='activation_function', values='final_loss').values
#ax.plot_surface(X, Y, Z, cmap=cmap, norm=norm)

# Set labels
#ax.set_xlabel('Activation Function')
#ax.set_ylabel('Loss Function')
ax.set_zlabel('lgo10(Final Loss)')

# Set title
ax.set_title(f'3D Plot of Activation Function vs Loss Function vs Final Loss (total square loss) {params}')

# Set ticks and labels using the mappings
ax.set_xticks(range(len(activation_mapping)))
ax.set_xticklabels([name for name, idx in sorted(activation_mapping.items(), key=lambda item: item[1])], rotation=90)

ax.set_yticks(range(len(loss_mapping)))
ax.set_yticklabels([name for name, idx in sorted(loss_mapping.items(), key=lambda item: item[1])], rotation=90)

# Set z-axis to logarithmic scale
#ax.set_zscale('log')

# set z range from 1 to 1.e-10
ax.set_zlim(-30., 6.)

# Set grid
ax.grid(True)

# highlight the point with the lowest final loss
min_loss_index = df['final_loss'].idxmin()
min_loss_row = df.iloc[min_loss_index]
ax.scatter(min_loss_row['activation_function'], min_loss_row['loss_function'], min_loss_row['final_loss'],
           color='red', s=100, label='Minimum Final Loss')
ax.text(min_loss_row['activation_function'], min_loss_row['loss_function'], min_loss_row['final_loss'],
        f'({list(activation_mapping.keys())[list(activation_mapping.values()).index(min_loss_row["activation_function"])]}, {list(loss_mapping.keys())[list(loss_mapping.values()).index(min_loss_row["loss_function"])]})', color='red')

# do the same for last 3 points with the lowest loss
min_loss_rows = df.nsmallest(3, 'final_loss')
for _, row in min_loss_rows.iterrows():
    ax.scatter(row['activation_function'], row['loss_function'], row['final_loss'],
               color='orange', s=100, label='Minimum Final Loss')
    ax.text(row['activation_function'], row['loss_function'], row['final_loss'],
            f'({list(activation_mapping.keys())[list(activation_mapping.values()).index(row["activation_function"])]}, {list(loss_mapping.keys())[list(loss_mapping.values()).index(row["loss_function"])]})', color='orange')

# tilt the view angle
ax.view_init(elev=20, azim=225)

# save as png
plt.savefig(f'3d_plot_{params}.png', dpi=300, bbox_inches='tight')

# Increase font size for title and labels
ax.title.set_fontsize(16)
ax.xaxis.label.set_fontsize(14)
ax.yaxis.label.set_fontsize(14)
ax.zaxis.label.set_fontsize(14)

# Increase font size for tick labels
ax.tick_params(axis='both', which='major', labelsize=12)
ax.tick_params(axis='z', which='major', labelsize=12)

# Show plot
plt.show()