#%%
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the data
data_path = '../plots/sizeexploration.txt'
df = pd.read_csv(data_path, sep="\t")

#%%
sns.set_theme(style="darkgrid")
# Create the plot with specified figure size
plt.figure(figsize=(6, 4))

# Plotting scatter with different colors and markers for each 'Input'
ax = sns.scatterplot(data=df, x='total kbyte', y='test/accuracy', hue='Input', style='Input', s=100)

# Add interpolated trend line for each 'Input' group, sorted by 'total kbyte' for smooth interpolation
for input_size in df['Input'].unique():
    subset = df[df['Input'] == input_size]
    subset = subset.sort_values('total kbyte')  # Sort by 'total kbyte'
    plt.plot(subset['total kbyte'], subset['test/accuracy'], linestyle='--', linewidth=1, label=f'Trend ({input_size})')

# Setting plot title and axis labels
plt.title('Test Accuracy vs Weight Memory Footprint')
plt.xlabel('Total memory footprint of weights (kbyte)')
plt.ylabel('Accuracy (%)')

# Setting the x-axis limits and ticks
ax.set_xlim(left=0, right=13)  # Set x-axis limit to 0 to 13
plt.xticks(range(0, 14, 1))  # Set x-axis ticks from 0 to 13 in steps of 1

# Display the legend and show the plot
plt.legend(title='Input Size')
plt.tight_layout()
plt.savefig('nnexploration.png', dpi=300)

plt.show()

# %%
