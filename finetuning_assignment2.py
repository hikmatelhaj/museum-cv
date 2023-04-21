
import numpy as np
def get_final_score(scores: list[float]) -> float:
    """
    Creating a weighted average of the scores
    """
    return np.average(scores, weights=[0, 0, 1])



# Read in histogram_scores.txt with 'with open' and 'readlines'
histogram_scores = []
with open("histogram_scores.txt", "r") as f:
    lines = f.readlines()
    for l in lines:
        histogram_scores.append(float(l))

matcher_scores = []
with open("matcher_scores.txt", "r") as f:
    lines = f.readlines()
    for l in lines:
        matcher_scores.append(float(l))
        
normalised_hog = []
with open("normalised_hog.txt", "r") as f:
    lines = f.readlines()
    for l in lines:
        normalised_hog.append(float(l))

final_scores = []
for i in range(len(histogram_scores)):
    
    final_score = get_final_score(np.array([matcher_scores[i], 1 - histogram_scores[i], 1 - normalised_hog[i]]))
    final_scores.append(final_score)

final_scores = np.array(final_scores)
# Get the indices of the 5 largest values
largest_indices = np.argpartition(final_scores, -5)[-5:]

# Extract the actual values using the indices
largest_values = final_scores[largest_indices]

print("avg is", np.average(final_scores))
print(largest_values)