import matplotlib.pyplot as plt
import numpy as np

# Data from the table
data = [
    {"config": "16-wide CNN, small fc", "width": 16, "size": 3.2, "accuracy": 98.92, "cycles": 686490, "time": 14.30, "type": "CNN"},
    {"config": "16-wide CNN", "width": 16, "size": 5.4, "accuracy": 99.06, "cycles": 785123, "time": 16.36, "type": "CNN"},
    {"config": "32-wide CNN", "width": 32, "size": 7.3, "accuracy": 99.28, "cycles": 1434667, "time": 29.89, "type": "CNN"},
    {"config": "48-wide CNN", "width": 48, "size": 9.3, "accuracy": 99.44, "cycles": 2083568, "time": 43.41, "type": "CNN"},
    {"config": "64-wide CNN", "width": 64, "size": 11.0, "accuracy": 99.55, "cycles": 2736250, "time": 57.01, "type": "CNN"},
    {"config": "4Bitsym FC", "width": None, "size": 12.3, "accuracy": 99.02, "cycles": 528377, "time": 11.01, "type": "4Bitsym"},
    {"config": "FP130 FC", "width": None, "size": 12.3, "accuracy": 98.86, "cycles": 481624, "time": 10.03, "type": "FP130"},
    {"config": "4Bitsym FC", "width": None, "size": 7.359375, "accuracy": 98.52, "cycles": None, "time": 6.59, "type": "4Bitsym"},
    {"config": "4Bitsym FC", "width": None, "size": 8.484375, "accuracy": 98.72, "cycles": None, "time": 7.60, "type": "4Bitsym"}
]

# Separate data by type
cnn_data = [d for d in data if d["type"] == "CNN"]
fp130_data = [d for d in data if d["type"] == "FP130"]
bitsym_data = [d for d in data if d["type"] == "4Bitsym"]

# Create figure with subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

# Plot 1: Time vs. Accuracy
ax1.scatter([d["time"] for d in cnn_data], [d["accuracy"] for d in cnn_data], 
           color='blue', s=100, alpha=0.7, label='CNN', marker='o')
ax1.scatter([d["time"] for d in fp130_data], [d["accuracy"] for d in fp130_data], 
           color='green', s=100, alpha=0.7, label='FP130', marker='^')
ax1.scatter([d["time"] for d in bitsym_data], [d["accuracy"] for d in bitsym_data], 
           color='red', s=100, alpha=0.7, label='4Bitsym', marker='s')

# Add labels for CNN points only
for d in cnn_data:
    if "small fc" in d["config"]:
        ax1.annotate('16-wide (small)', (d["time"], d["accuracy"]), 
                    xytext=(5, 5), textcoords='offset points', fontsize=9)
    else:
        ax1.annotate(f'{d["width"]}-wide', (d["time"], d["accuracy"]), 
                    xytext=(5, 5), textcoords='offset points', fontsize=9)

ax1.set_xlabel('Time (ms)')
ax1.set_ylabel('Test Accuracy (%)')
ax1.set_title('Time vs. Accuracy Trade-off')
ax1.grid(True, alpha=0.3)
ax1.legend()

# Plot 2: Size vs. Accuracy
ax2.scatter([d["size"] for d in cnn_data], [d["accuracy"] for d in cnn_data], 
           color='blue', s=100, alpha=0.7, label='CNN', marker='o')
ax2.scatter([d["size"] for d in fp130_data], [d["accuracy"] for d in fp130_data], 
           color='green', s=100, alpha=0.7, label='FP130', marker='^')
ax2.scatter([d["size"] for d in bitsym_data], [d["accuracy"] for d in bitsym_data], 
           color='red', s=100, alpha=0.7, label='4Bitsym', marker='s')

# Add labels for CNN points only
for d in cnn_data:
    if "small fc" in d["config"]:
        ax2.annotate('16-wide (small)', (d["size"], d["accuracy"]), 
                    xytext=(5, 5), textcoords='offset points', fontsize=9)
    else:
        ax2.annotate(f'{d["width"]}-wide', (d["size"], d["accuracy"]), 
                    xytext=(5, 5), textcoords='offset points', fontsize=9)

ax2.set_xlabel('Size (kB)')
ax2.set_ylabel('Test Accuracy (%)')
ax2.set_title('Size vs. Accuracy Trade-off')
ax2.grid(True, alpha=0.3)
ax2.legend()

# Adjust layout and save
plt.tight_layout()
plt.savefig('cnn_tradeoff_plots.png', dpi=300, bbox_inches='tight')
plt.show()

print("Trade-off plots created and saved as 'tradeoff_plots.png'")