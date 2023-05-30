"""
This is code to test out if the hidden markov model "spread" out in case the emission probabilities are uniform.
"""

import numpy as np
import pandas as pd
from hmmlearn import hmm
import transition_matrix as tm
import sys, os
sys.path.append(os.path.abspath(os.path.join('.')))
from geomap import showHeatmap

emission_probability = []
states = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12", "13", "14", "15", "16", "17", "18", "19", "RI", "II", "V"]
n_states = len(states)
# for zaal in states:
#     if zaal == detected_zaal:
#         emission_probability.append(chances_TP)
#     else:
#         emission_probability.append(chances_FP)

emission_probability = np.empty((len(states), len(states))); emission_probability.fill(1/len(states))
start_probs = np.zeros(len(states)); start_probs[0] = 1
print(start_probs)
# emission_probability = np.array(emission_probability)
# print("\nEmission probability:\n", emission_probability)

observations_sequence = np.array([4]).reshape(-1, 1)
# print("observation sequence:", observations_sequence)
model = hmm.CategoricalHMM(n_components=n_states)
model.startprob_ = start_probs
model.transmat_ = tm.transition_matrix
model.emissionprob_ = emission_probability
# hidden_states = model.predict(observations_sequence)
hidden_states_probs = model.predict_proba(observations_sequence)
print(hidden_states_probs)
# for el in hidden_states_probs:
#     print(el)
#     data = {"Hall": states, "probability": np.squeeze(el)}
#     df = pd.DataFrame(data)
#     showHeatmap(df)

# while True:
#     obs = np.array([2]).reshape(-1, 1)
#     model = hmm.CategoricalHMM(n_components=n_states)
#     model.startprob_ = start_probs
#     model.transmat_ = tm.transition_matrix
#     model.emissionprob_ = emission_probability
#     model.fit(obs)
#     start_probs = np.squeeze(model.predict_proba(obs))
#     print(start_probs)
#     data = {"Hall": states, "probability": start_probs}
#     df = pd.DataFrame(data)
#     showHeatmap(df)