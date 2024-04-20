#%%
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

# Load your data
# Assuming df is already loaded as shown earlier
# df = pd.read_csv('your_data_file.csv', sep='\t')
df = pd.read_csv('clean_30epruns.txt', sep='\t')

# Preparing the figure
fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharey=True)

# Defining color palette and markers
palette = sns.color_palette('tab10', n_colors=len(df['QuantType'].unique()))
marker_dict = {quant_type: marker for quant_type, marker in zip(df['QuantType'].unique(), ['o', 's', 'D', '^', 'v', '<', '>', 'p', '*', '+'])}

for i, quant_type in enumerate(df['QuantType'].unique()):
    # Filter data for each quantization type
    temp_df = df[df['QuantType'] == quant_type].copy()
    
    # Calculate Error Rates as 100 - Accuracy
    temp_df['Error Rate/train'] = 100 - temp_df['Accuracy/train']
    temp_df['Error Rate/test'] = 100 - temp_df['Accuracy/test']
    
    # Plot Training Error Rate
    axes[0].scatter(temp_df['Totalbits'], temp_df['Error Rate/train'], color=palette[i], label=quant_type, marker=marker_dict[quant_type])
    
    # Plot Test Error Rate
    axes[1].scatter(temp_df['Totalbits'], temp_df['Error Rate/test'], color=palette[i], marker=marker_dict[quant_type])

# Setting log scale for x-axis and y-axis
axes[0].set_xscale('log')
axes[0].set_yscale('log')
axes[1].set_xscale('log')

# Adding titles and labels
axes[0].set_title('Training Error Rate vs. Total Bits', fontsize=14)
axes[0].set_xlabel('Total Bits (log scale)', fontsize=12)
axes[0].set_ylabel('Error Rate (%)', fontsize=12)

axes[1].set_title('Test Error Rate vs. Total Bits', fontsize=14)
axes[1].set_xlabel('Total Bits (log scale)', fontsize=12)

# Adding legend to the first plot
axes[0].legend(title='Quantization Type', bbox_to_anchor=(1.05, 1), loc='upper left')

plt.tight_layout()
plt.show()

# %%
