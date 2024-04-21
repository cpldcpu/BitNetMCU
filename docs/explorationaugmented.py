import matplotlib.pyplot as plt
import pandas as pd

# Data
data = {
    "w1": [64, 64, 64, 56, 72, 80, 96, 48, 48, 72, 72, 64, 64, 64, 56, 48, 48, 40, 40, 40, 40, 40],
    "w2": [64, 64, 48, 56, 80, 80, 96, 48, 48, 64, 72, 96, 128, 72, 112, 160, 128, 64, 40, 40, 40, 40],
    "w3": [64, 0, 32, 56, 0, 80, 0, 0, 48, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    "BPW": [4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 8, 4, 5, 2],
    "total_kbyte": [12.3125, 10.3125, 10.40625, 10.3359375, 12.203125, 16.640625, 16.96875, 7.359375,
                    8.484375, 11.5625, 11.8828125, 11.46875, 12.625, 10.6015625, 10.609375, 10.53125,
                    9.625, 6.5625, 11.953125, 5.976563, 7.460703, 2.988281],
    "Layers": [3, 2, 3, 3, 2, 3, 2, 2, 3, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
    "test_accuracy": [99.01, 98.79, 98.51, 98.73, 98.98, 99.04, 98.98, 98.52, 98.72, 98.85, 98.94, 98.96,
                      99.03, 98.76, 98.87, 98.79, 98.7, 98.29, 98.6, 98.13, 98.44, 96.79]
}

df = pd.DataFrame(data)

# Filter out a specific data point
df_filtered = df[~((df['w1'] == 40) & (df['w2'] == 40) & (df['w3'] == 0) & (df['BPW'] == 2))]

# Group the filtered data by the number of layers
groups_filtered = df_filtered.groupby('Layers')

# Colors and markers
colors = {2: 'blue', 3: 'green'}
markers = {2: 'o', 3: 's'}  # Circle for 2 layers, square for 3 layers

# Plotting
fig, ax = plt.subplots(figsize=(8, 6))

for name, group in groups_filtered:
    ax.scatter(group['total_kbyte'], group['test_accuracy'], c=colors[name], label=labels[name],
               marker=markers[name], s=100, alpha=0.6, edgecolors='none')
    for i, row in group.iterrows():
        label = f"{int(row['w1'])}/{int(row['w2'])}/{int(row['w3'])}/{int(row['BPW'])}b"
        # Bold specific labels
        if label == "40/40/0/8b" or label == "64/48/32/4b":
            ax.text(row['total_kbyte'] + 0.1, row['test_accuracy'], label, fontsize=8, fontweight='bold',
                    verticalalignment='center', horizontalalignment='left')
        else:
            ax.text(row['total_kbyte'] + 0.1, row['test_accuracy'], label, fontsize=8, verticalalignment='center',
                    horizontalalignment='left')

ax.set_xlabel('Total kbyte')
