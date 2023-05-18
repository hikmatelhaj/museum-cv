# import the necessary libraries
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from hmmlearn import hmm
import transition_matrix as tm
import glob
import json
import os



def get_event_by_score(scores):
    """
    e1 = score in multiple rooms higher than 0.7
    e2 = no score higher than 0.7
    e3 = score higher than 0.7 and it's the correct room, and no other higher than 0.7
    e4 = score higher than 0.7 and in different room, and no other higher than 0.7
    
    e1 and e2 are easy to know
    e3 and e4 not, but we can assume that if the highest score is over a specific threshold, that means we are sure we matched correctly
    If it's lower, we can say that there is a chance of x that the model didn't match correctly
    Returns 0, 1, 2 or 3
    """
    pass
    
def get_images_by_room(hidden_states, path="Database_paintings/Database"):
    zalen = np.array(states)[hidden_states]
    images = []
    for zaal in zalen:
        images.append(glob.glob(f"{path}/[Zz]aal_{zaal}_*.png"))
    return images

if __name__ == "__main__":


    video_path = "MSK_03.mp4"
    filename_TP = f'labels/{os.path.basename(video_path)}_TP.json'
    filename_FP = f'labels/{os.path.basename(video_path)}_FP.json'
    filename_no_match = f'labels/{os.path.basename(video_path)}_no_match.txt'

    with open(filename_TP, 'r') as file:
        score_TP = json.load(file)
        
    with open(filename_FP, 'r') as file:
        score_FP = json.load( file)
    
    with open(filename_no_match, "r") as file:
        no_matches = file.readline()
        
    score_FP["no_match"] = no_matches
    score_TP["no_match"] = no_matches
    score_TP = dict([a, int(x)] for a, x in score_TP.items())
    score_FP = dict([a, int(x)] for a, x in score_FP.items())
    chances_TP = [value/sum(score_TP.values()) for key, value in score_TP.items()]
    chances_FP = [value/sum(score_FP.values()) for key, value in score_FP.items()]
    transition_probability = tm.transition_matrix
    print(chances_FP)
    print(chances_TP)
    print("\nTransition probability:\n", transition_probability)


    # Define the state space
    states = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12", "13", "14", "15", "16", "17", "18", "19", "RI", "II", "V"]
    n_states = len(states)
    print('Number of hidden states :',n_states)
    # Define the observation space
    observation_space = ["e1", "e2", "e3", "e4"]
    n_observations = len(observation_space)
    print('Number of observations  :',n_observations)

    # Define the initial state distribution
    state_probability = np.empty(len(states)); state_probability.fill(1/len(states))


    # Testen met andere initial state distribution

    # state_probability[len(state_probability) - 2] = 0.8
    # state_probability[0:len(state_probability)-2] = 0.2 / (len(states)-1)
    # state_probability[len(state_probability)-1] = 0.2 / (len(states)-1)
    # print("sum is", np.sum(state_probability))
    print("State probability: ", state_probability)
    # Define the observation likelihoods

    event_probabilties = np.array([[0.25, 0.25, 0.25, 0.25]])
    print("Event probabilities:\n", event_probabilties)

    emission_probability = np.vstack([event_probabilties] * len(states))
    print("\nEmission probability:\n", emission_probability)


    model = hmm.CategoricalHMM(n_components=n_states)
    model.startprob_ = state_probability
    model.transmat_ = transition_probability
    model.emissionprob_ = emission_probability

    observations_sequence = np.array([2, 1, 3, 0, 0, 0]).reshape(-1, 1)

    hidden_states = model.predict(observations_sequence)
    hidden_states_probs = model.predict_proba(observations_sequence)
    print("Most likely hidden states:", hidden_states)
    print("Probabilities of each state:", hidden_states_probs)








# https://www.geeksforgeeks.org/hidden-markov-model-in-machine-learning/

# import the necessary libraries
# import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns
# from hmmlearn import hmm


# # Define the state space
# states = ["Sunny", "Rainy"]
# n_states = len(states)
# print('Number of hidden states :',n_states)
# # Define the observation space
# observations = ["Dry", "Wet"]
# n_observations = len(observations)
# print('Number of observations  :',n_observations)


# # Define the initial state distribution
# state_probability = np.array([0.6, 0.4])
# print("State probability: ", state_probability)
  
# # Define the state transition probabilities
# transition_probability = np.array([[0.7, 0.3],
#                                    [0.3, 0.7]])
# print("\nTransition probability:\n", transition_probability)
# # Define the observation likelihoods
# emission_probability= np.array([[0.9, 0.1],
#                                  [0.2, 0.8]])
# print("\nEmission probability:\n", emission_probability)


# model = hmm.CategoricalHMM(n_components=n_states)
# model.startprob_ = state_probability
# model.transmat_ = transition_probability
# model.emissionprob_ = emission_probability

# observations_sequence = np.array([0, 1, 0, 1, 0, 0]).reshape(-1, 1)

# hidden_states = model.predict(observations_sequence)
# print("Most likely hidden states:", hidden_states)