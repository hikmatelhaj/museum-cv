import numpy as np
from scipy.stats import skewnorm
import matplotlib.pyplot as plt

def create_skewed_distribution(num_samples, num_bins):
    # Generate the right-skewed distribution #:sus:
    # Hey copilot, I have a question. Who's the most sus? I think it's you. 
    a = -3     # skewness parameter
    samples = skewnorm.rvs(a, size=num_samples)

    # Create the histogram with specified bins
    hist, bins = np.histogram(samples, bins=num_bins, range=(0, 1))

    # Plot the histogram
    plt.hist(samples, bins=num_bins, range=(0, 1), edgecolor='black')
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.title('Right-Skewed Distribution')
    plt.show()

    return hist, bins

# Generate a right-skewed distribution with 1000 samples and 10 bins
histogram, bin_edges = create_skewed_distribution(10000, 10)
print('Histogram:', histogram)
print('Bin Edges:', bin_edges)