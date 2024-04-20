#%%
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Assuming df is your DataFrame loaded from the file
# Replace the loading part with your actual DataFrame loading code
df = pd.read_csv('clean_30epruns.txt', sep='\t')

#%%
# Plotting training and test loss in separate panes, horizontally, in the same figure
fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharey=True)  # Using shared y-axis

# Plotting Training Loss on the first pane
sns.scatterplot(ax=axes[0], data=df, x='Totalbits', y='Loss/train', hue='QuantType', style='QuantType', palette='tab10', s=100, edgecolor='black', linewidth=0.5)
axes[0].set_title('Training Loss vs. Total Bits', fontsize=14)
axes[0].set_xlabel('Total Bits (log scale)', fontsize=12)
axes[0].set_ylabel('Loss (log scale)', fontsize=12)
axes[0].set_xscale('log')
axes[0].set_yscale('log')
axes[0].legend(title='Quantization Type', bbox_to_anchor=(1.05, 1), loc='upper left')

# Plotting Test Loss on the second pane
sns.scatterplot(ax=axes[1], data=df, x='Totalbits', y='Loss/test', hue='QuantType', style='QuantType', palette='tab10', s=100, edgecolor='black', linewidth=0.5)
axes[1].set_title('Test Loss vs. Total Bits', fontsize=14)
axes[1].set_xlabel('Total Bits (log scale)', fontsize=12)
# No need for y-label as it shares with the first pane
axes[1].set_xscale('log')
axes[1].set_yscale('log')
axes[1].legend(title='Quantization Type', bbox_to_anchor=(1.05, 1), loc='upper left')

# Adjusting layout for shared y-axis and ensuring the legend is not cut off
plt.tight_layout()

# Displaying the plot
plt.show()

# %%
