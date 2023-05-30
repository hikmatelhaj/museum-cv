import matplotlib.pyplot as plt
import json
import numpy as np

# Load the data from the JSON file
filename_TP = 'labels/final_TP.json'
filename_FP = 'labels/final_FP.json'

with open(filename_TP, 'r') as file:
    score_TP = json.load(file)

with open(filename_FP, 'r') as file:
    score_FP = json.load(file)

# Extract the keys and values from the dictionary
data_TP = []
for key, value in score_TP.items():
    for i in range(int(value)):
        data_TP.append(float(key))

data_FP = []
for key, value in score_FP.items():
    for i in range(int(value)):
        data_FP.append(float(key))

# Adjust the binning
num_bins_TP = len(score_TP.keys())
num_bins_FP = len(score_FP.keys())

# Plot the histograms
plt.hist(data_FP, bins=num_bins_FP, color='black', alpha=0.6, label='False positives', align='mid')
plt.hist(data_TP, bins=num_bins_TP, alpha=0.6, label='True positives', align='mid')

# Add labels and title
plt.xlabel('Score')
plt.ylabel('Frequency')
plt.title('Distribution of True Positives and False Positives')
plt.xticks(np.arange(0, 1, 0.1))
# Add a legend
plt.legend()
# plt.savefig("plot.png", bbox_inches='tight', dpi=300)
# Display the plot
plt.show()