import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from hmmlearn import hmm

import sys, os
sys.path.append(os.path.abspath(os.path.join('.')))
sys.path.append(os.path.abspath(os.path.join('assignment3/')))
import transition_matrix as tm
from geomap import showHeatmap
import glob
import json
import os
import cv2, time, math
import pandas as pd

def calculate_hmm(detected_score, detected_zaal, chances_FP, chances_TP, start_probs, prev_obs):
    """
    detected_score: the score of the detected painting
    detected_zaal: the room of the database match of the detected painting
    chances_FP: the chances of a false positive, this is fixed
    chances_TP: the chances of a true positive, this is fixed
    start_probs: the start probabilities of the HMM
    prev_obs: the previous observation
    """
    observation = translation_event[str(detected_score)] # converted to an observation
    emission_probability = []

    if detected_zaal == "no_match":
        emission_probability = np.empty((len(states), len(states))); emission_probability.fill(1/len(states))
    else:
        for zaal in states:
            if zaal == detected_zaal:
                emission_probability.append(chances_TP)
            else:
                emission_probability.append(chances_FP)
    
        emission_probability = np.array(emission_probability)
    # print("\nEmission probability:\n", emission_probability)
    observations_sequence = np.array([translation_event[str(prev_obs)], observation]).reshape(-1, 1)
    # print("observation sequence:", observations_sequence)
    model = hmm.CategoricalHMM(n_components=n_states)
    model.startprob_ = start_probs
    model.transmat_ = tm.transition_matrix
    model.emissionprob_ = emission_probability
    hidden_states_probs = model.predict_proba(observations_sequence)[1]
    # print("Most likely hidden states:", hidden_states, "in zaal", states[hidden_states[0]])
    # print("Probabilities of each state:", hidden_states_probs)
    data = {"Hall": states, "probability": np.squeeze(hidden_states_probs)}
    df = pd.DataFrame(data)
    return np.squeeze(hidden_states_probs), states[np.argmax(hidden_states_probs)], df, np.max(hidden_states_probs)


if __name__ == "__main__":


    filename_TP = f'labels/final_TP.json'
    filename_FP = f'labels/final_FP.json'

    with open(filename_TP, 'r') as file:
        score_TP = json.load(file)
        
    with open(filename_FP, 'r') as file:
        score_FP = json.load(file)

    score_TP = dict([a, int(x)] for a, x in score_TP.items())
    score_FP = dict([a, int(x)] for a, x in score_FP.items())
    chances_TP = [value/sum(score_TP.values()) for key, value in score_TP.items()]
    chances_FP = [value/sum(score_FP.values()) for key, value in score_FP.items()]
    states = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12", "13", "14", "15", "16", "17", "18", "19", "RI", "II", "V"]

    n_states = len(states)
    translation_event = {"0.0": 0, "0.1": 1, "0.2": 2, "0.3": 3, "0.4": 4, "0.5": 5, "0.6": 6, "0.7": 7, "0.8": 8, "0.9": 9, "no_match": 0}
    transition_probability = tm.transition_matrix
    
    # Define the initial state distribution: every state is equally likely
    state_probability = np.zeros(len(states))
    state_probability[0] = 1
    last_probabilities = state_probability
    data = {"Hall": states, "probability": np.squeeze(last_probabilities)}
    df = pd.DataFrame(data)
    showHeatmap(df)

    for i in range(10):
        last_last_probabilities = last_probabilities
        last_probabilities, zaal_predict, df, percentage = calculate_hmm("no_match", "no_match", chances_FP, chances_TP, last_last_probabilities, "no_match")
        showHeatmap(df)
    
    last_probabilities, zaal_predict, df, percentage = calculate_hmm("0.9", "19", chances_FP, chances_TP, last_last_probabilities, "0.9")
    showHeatmap(df)