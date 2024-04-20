#%%
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Read the data
data_path = '12kruns.txt'
data = pd.read_csv(data_path, delimiter=",", skiprows=1, header=None)

#%%
# Assign column headers (from first row of the original file)
headers = pd.read_csv(data_path, nrows=0).columns.tolist()
data.columns = headers

# Extract schedule from runname
data['Schedule'] = data['runname'].str.extract(r'(Lin|Cos)')

# Function to categorize groups with descriptive names
def categorize_group(row):
    if row['Schedule'] == 'Lin' and row['num_epochs'] == 30:
        return 'A: L_30ep'
    elif row['Schedule'] == 'Cos' and row['num_epochs'] == 30:
        return 'B: C_30ep'
    elif row['Schedule'] == 'Cos' and row['num_epochs'] == 60:
        return 'C: C_60ep'
    elif row['Schedule'] == 'Cos' and row['num_epochs'] == 120:
        return 'D: C_120ep'
    else:
        return 'Other'  # To catch any cases that don't match expected conditions

# Applying the categorization function
data['Group'] = data.apply(categorize_group, axis=1)

# Ensuring groups are ordered correctly for plotting
group_order = ['A: L_30ep', 'B: C_30ep', 'C: C_60ep', 'D: C_120ep']
data['Group'] = pd.Categorical(data['Group'], categories=group_order, ordered=True)

# Set up the plotting
sns.set(style="whitegrid")
fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(10, 6))

# Training Accuracy
sns.lineplot(ax=axes[0, 0], data=data, x='Group', y='Accuracy/train', hue='QuantType', style='QuantType', markers=True, dashes=False, palette='tab10', markersize=10)
axes[0, 0].set_title('Training Accuracy by Group and QuantType')
axes[0, 0].set_xlabel('Group')
axes[0, 0].set_ylabel('Train Accuracy (%)')

# Training Loss
sns.lineplot(ax=axes[0, 1], data=data, x='Group', y='Loss/train', hue='QuantType', style='QuantType', markers=True, dashes=False, palette='tab10', markersize=10)
axes[0, 1].set_yscale('log')
axes[0, 1].set_title('Training Loss by Group and QuantType')
axes[0, 1].set_xlabel('Group')
axes[0, 1].set_ylabel('Train Loss (log scale)')

# Test Accuracy
sns.lineplot(ax=axes[1, 0], data=data, x='Group', y='Accuracy/test', hue='QuantType', style='QuantType', markers=True, dashes=False, palette='tab10', markersize=10)
axes[1, 0].set_title('Test Accuracy by Group and QuantType')
axes[1, 0].set_xlabel('Group')
axes[1, 0].set_ylabel('Test Accuracy (%)')

# Test Loss
sns.lineplot(ax=axes[1, 1], data=data, x='Group', y='Loss/test', hue='QuantType', style='QuantType', markers=True, dashes=False, palette='tab10', markersize=10)
axes[1, 1].set_yscale('log')
axes[1, 1].set_title('Test Loss by Group and QuantType')
axes[1, 1].set_xlabel('Group')
axes[1, 1].set_ylabel('Test Loss (log scale)')

plt.tight_layout()
plt.show()

# %%
