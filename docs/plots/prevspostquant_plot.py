#%%
import matplotlib.pyplot as plt
import pandas as pd

# Data preparation
data = {
    "Quantization during training": [
        "FP32", "FP32", "FP32", "FP32", "FP32", "FP32",
        "QAT", "QAT", "QAT", "QAT", "QAT",
        "4 Bit", "4 Bit", "4 Bit", "4 Bit",
        "8 Bit", "8 Bit", "8 Bit", "8 Bit", "8 Bit"
    ],
    "Postquantization": [
        1, 1.6, 2, 4, 8, None,
        1, 1.6, 2, 4, 8,
        1, 1.6, 2, 4,
        1, 1.6, 2, 4, 8
    ],
    "Test/Accuracy": [
        51.74, 69.96, 80.09, 85.39, 97.92, 98.22,
        97.03, 97.39, 97.68, 97.92, 98.02,
        70.14, 91.99, 96.63, 97.92,
        59.8, 83.26, 87.8, 94.34, 98.02
    ]
}
df = pd.DataFrame(data)
#%%

# Drop rows with NaN in 'Postquantization'
df_clean = df.dropna(subset=['Postquantization'])

# Plotting setup
plt.figure(figsize=(8, 5))
markers = ['o', 's', '^', 'D', 'x']  # Different symbols
colors = ['blue', 'green', 'red', 'purple', 'orange']  # Different colors

quantization_categories = df_clean['Quantization during training'].unique()

for i, category in enumerate(quantization_categories):
    subset = df_clean[df_clean['Quantization during training'] == category]
    plt.plot(subset['Postquantization'], subset['Test/Accuracy'], marker=markers[i], color=colors[i], label=category, linestyle='-', markersize=8)

plt.xlabel('Postquantization [bits]', fontsize=12)
plt.ylabel('Test/Accuracy [%]', fontsize=12)
plt.title('Test Accuracy vs Postquantization, Grouped by Quantization during Training', fontsize=14)
plt.legend(title='Quantization during training')
plt.grid(True, which='both', linestyle='--', linewidth=0.5)

plt.show()

# %%
