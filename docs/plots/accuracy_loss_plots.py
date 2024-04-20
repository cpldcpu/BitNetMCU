#%%
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the data
df = pd.read_csv('clean_30epruns.txt', sep='\t')

#%%

# Filter out '8bit' and 'None' quantization types
df_filtered = df[~df['QuantType'].isin(['8bit', 'None'])]

# Prepare combined data for accuracy plot
train_data = df_filtered[['Totalbits', 'Accuracy/train', 'Loss/train']].rename(columns={'Accuracy/train': 'Accuracy', 'Loss/train': 'Loss'})
train_data['Type'] = 'Train'
test_data = df_filtered[['Totalbits', 'Accuracy/test', 'Loss/test']].rename(columns={'Accuracy/test': 'Accuracy', 'Loss/test': 'Loss'})
test_data['Type'] = 'Test'
combined_data = pd.concat([train_data, test_data])

# Prepare combined data for loss plot
train_loss_data = df_filtered[['Totalbits', 'Loss/train']].rename(columns={'Loss/train': 'Loss'})
train_loss_data['Type'] = 'Train'
test_loss_data = df_filtered[['Totalbits', 'Loss/test']].rename(columns={'Loss/test': 'Loss'})
test_loss_data['Type'] = 'Test'
combined_loss_data = pd.concat([train_loss_data, test_loss_data])

# Plotting Accuracy
plt.figure(figsize=(8, 4))
sns.scatterplot(data=combined_data, x='Totalbits', y='Accuracy', hue='Type', style='Type', palette='Set1', s=100, edgecolor='black', linewidth=0.5)
position_bits = 8 * 12 * 1024  # Corrected position
plt.axvline(x=position_bits, color='r', linestyle='--')
plt.text(position_bits, combined_data['Accuracy'].min(), '12kbyte', color='r', ha='right')
plt.title('Train vs. Test Accuracy vs. Total Bits', fontsize=14)
plt.xlabel('Total Bits (log scale)', fontsize=12)
plt.ylabel('Accuracy', fontsize=12)
plt.xscale('log')
plt.legend(title='Data Type', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()

# Plotting Loss
plt.figure(figsize=(8, 4))
sns.scatterplot(data=combined_loss_data, x='Totalbits', y='Loss', hue='Type', style='Type', palette='Set1', s=100, edgecolor='black', linewidth=0.5)
plt.axvline(x=position_bits, color='r', linestyle='--')
plt.text(position_bits, combined_loss_data['Loss'].min(), '12kbyte', color='r', ha='right')
plt.title('Train vs. Test Loss vs. Total Bits', fontsize=14)
plt.xlabel('Total Bits (log scale)', fontsize=12)
plt.ylabel('Loss', fontsize=12)
plt.xscale('log')
plt.yscale('log')
plt.legend(title='Data Type', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()

# %%
