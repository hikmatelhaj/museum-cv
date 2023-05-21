# import the necessary libraries
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from hmmlearn import hmm
import transition_matrix as tm
import glob
import json
import os
import cv2, time, math
import pandas as pd

import sys, os
sys.path.append(os.path.abspath(os.path.join('.')))
import assignment1
from assignment2.assignment2 import *
from geomap import showHeatmap
from Extractor import *

states = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12", "13", "14", "15", "16", "17", "18", "19", "RI", "II", "V"]


def video_frame_process(video_path, state_probability):
    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    last_probabilities = state_probability
    f = 0
    no_matched_found = 0
    total_seconds = 0 # keep track of the total frames that was processed (can be converted to secs by dividing by fps)
    last_decile = 0 # keep track of the last 10 seconds that was processed
    scores_per_decile = {}
    st = time.time()
    found_observations = []
    
    extractor = Extractor()
    
    seconds_to_wait = 1 # wait x seconds before processing the next frame after a sharp frame
    while True:
        process = False
        total_seconds += 1
        # if total_seconds % 2 != 0:
        #     continue
        
        flag, frame = cap.read()
        if frame is None: # if video is over
            break
        if f < seconds_to_wait * fps:
            f += 1
        if f % (seconds_to_wait * fps) == 0:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            lol = cv2.Laplacian(gray, cv2.CV_64F).var() 
            if lol > 30:
                process = True
                # Frame is not blurry
            else:
                continue

        # wait until next sharp frame ( min 30 frames in between processing of sharp frames)
        if process and f % (seconds_to_wait * fps) == 0:
            f = 1
            process = False
            # print("processing frame")
            second = total_seconds / fps # get the current second in the video
            decile = math.floor(second / 10) * 10 # round to the nearest decile
            if decile not in scores_per_decile.keys():
                scores_per_decile[decile] = []
            results = extractor.extract_and_crop_image(frame, False) # assignment1.process_single_image(frame, False)
            for idx, extracted_painting in enumerate(results):
                if idx > 2: # process max 2 paintings per frame
                    continue
                # cv2.imwrite(f'./dataset_video/painting_{idx}', extracted_painting)
                scores, files = calculate_score_assignment2_multi(extracted_painting, "Database_paintings/Database")
                
                canvas = np.zeros((600, 1000, 3), dtype=np.uint8)
                
                scores = np.array(scores)
                ind = np.argpartition(scores, -2)[-2:]
                top2 = scores[ind]
                files = np.array(files)
                files = files[ind]
                scores_per_decile[decile].append({"score": top2[1], "file": files[1], "extracted": extracted_painting, "to_check": frame})
                if last_decile == decile: # only if a new decile is reached, show the best results
                    continue
                
                items = scores_per_decile.get(last_decile, [])
                et = time.time()
                elapsed_time = et - st
                print('It took', elapsed_time, 'seconds to process the last', decile - last_decile ,'seconds')
                
                    
                if len(items) == 0:
                    # hmm calculate
                    found_observations.append("no_match")
                    try:
                        prev = found_observations[-2]
                    except:
                        prev = found_observations[-1]
                        
                    last_probabilities, zaal_predict, df = calculate_hmm("no_match", "no_match", chances_FP, chances_TP, last_probabilities, prev)
                
                    # calculate_hmm_full_path(found_observations, zaal, chances_FP, chances_TP)
                    print("No paintings detected in the last", decile - last_decile , "seconds")
                    print("Predicted to be in zaal", zaal_predict)
                    showHeatmap(df)
                    scores_per_decile[decile] = []
                    last_decile = decile # update the last decile
                    st = time.time()
                    continue
                print("Currently in", decile, "seconds")
                amount_of_deciles_without_paintings = ((decile - last_decile) // 10) - 1 # -1 because the last 10 secs a painting was detected, otherwise the code wouldn't reach here
                no_matched_found += amount_of_deciles_without_paintings
                    
                # Find the item with the highest score
                highest_score_item = max(items, key=lambda x: x["score"])

                # Retrieve the desired information from the highest score item
                highest_score = highest_score_item["score"]
                file_name = highest_score_item["file"]
                highest_extracted_painting = highest_score_item["extracted"]
                highest_to_check = highest_score_item["to_check"]

                
                
                # clear to save space
                scores_per_decile[decile] = []
                
                # process next decile
                last_decile = decile # update the last decile
                
                score_bin = math.floor(highest_score * 10) / 10
                if score_bin == 1.0:
                    score_bin = 0.9

                # hmm calculate
                zaal = get_zaal_by_filename(file_name)
                found_observations.append(score_bin)
                try:
                    prev = found_observations[-2]
                except:
                    prev = found_observations[-1]
                last_probabilities, zaal_predict, df = calculate_hmm(score_bin, zaal, chances_FP, chances_TP, last_probabilities, prev)
                
                print("Predicted", zaal_predict)
                print("db match zaal", zaal)
                # calculate_hmm_full_path(found_observations, zaal, chances_FP, chances_TP)
                st = time.time()
                
                
                img2 = cv2.imread("Database_paintings/Database/" + file_name) # DB foto highest
                img2 = cv2.GaussianBlur(img2, (5, 5), 0)
                img3 = highest_extracted_painting # extracted

                img2 = cv2.resize(img2, (250, 250)) 
                img3 = cv2.resize(img3, (250, 250))
                
                # img3 = cv2.resize(img3, (350, 250))

                canvas = np.zeros((600, 800, 3), dtype=np.uint8)

                img2 = cv2.resize(img2, (250, 250))

                canvas[50:300, 50:300] = img3
                cv2.putText(canvas, f"Extracted", (275, 650), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

                canvas[50:300, 350:600] = img2
                cv2.putText(canvas, f"DB highest {round(highest_score, 2)}", (400, 350), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

                highest_to_check = cv2.resize(highest_to_check, (350, 250)) #  into shape (250,350,3)
                canvas[350:600, 225:575] = highest_to_check


                cv2.imshow("Display Images", canvas)
                key = cv2.waitKey(0)
                # Check the pressed key and do something based on it
                while key != ord('y') and key != ord('n'):
                    key = cv2.waitKey(0)
                cv2.destroyAllWindows()
                showHeatmap(df)
            
def get_zaal_by_filename(filename):
    filename = filename.split("_")
    return filename[1]
    
def get_images_by_room(hidden_states, path="Database_paintings/Database"):
    zalen = np.array(states)[hidden_states]
    images = []
    for zaal in zalen:
        images.append(glob.glob(f"{path}/[Zz]aal_{zaal}_*.png"))
    return images

def calculate_hmm(detected_score, detected_zaal, chances_FP, chances_TP, start_probs, prev_obs):
    observation = translation_event[str(detected_score)]
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
    hidden_states = model.predict(observations_sequence)
    hidden_states_probs = model.predict_proba(observations_sequence)[1]
    # print("Most likely hidden states:", hidden_states, "in zaal", states[hidden_states[0]])
    # print("Probabilities of each state:", hidden_states_probs)
    
    data = {"Hall": states, "probability": np.squeeze(hidden_states_probs)}
    df = pd.DataFrame(data)
    return np.squeeze(hidden_states_probs), states[hidden_states[0]], df
      
      
def calculate_hmm_full_path(detected_scores, detected_zaal, chances_FP, chances_TP):
    observations = []
    for score in detected_scores:
        observations.append(translation_event[str(score)])
    emission_probability = []
    state_probability = np.empty(len(states)); state_probability.fill(1/len(states))
    for zaal in states:
        if zaal == detected_zaal:
            emission_probability.append(chances_TP)
        else:
            emission_probability.append(chances_FP)
    
    emission_probability = np.array(emission_probability)
    # print("\nEmission probability:\n", emission_probability)

    observations_sequence = np.array([observations]).reshape(-1, 1)
    model = hmm.CategoricalHMM(n_components=n_states)
    
    model.startprob_ = state_probability
    model.transmat_ = tm.transition_matrix
    model.emissionprob_ = emission_probability
    hidden_states = model.predict(observations_sequence)
    hidden_states_probs = model.predict_proba(observations_sequence)
    states_np = np.array(states)
    print("Full path, zalen zijn", states_np[hidden_states])
    # print("Probabilities of each state:", hidden_states_probs)
    
    # data = {"State": states, "Probability": hidden_states_probs}
    # df = pd.DataFrame(data)
    # showHeatmap(df)
    
    
    return hidden_states
          
if __name__ == "__main__":


    filename_TP = f'labels/final_TP.json'
    filename_FP = f'labels/final_FP.json'
    filename_no_match = f'labels/final_no_match.txt'

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
    
    print(chances_TP)
    print(chances_FP)

    # Define the initial state distribution

    # Define the state space
    n_states = len(states)
    # print('Number of hidden states :',n_states)
    # Define the observation space
    # observation_space = ["e1", "e2", "e3", "e4"]
    n_observations = 11

    # print('Number of observations  :',n_observations)
    translation_event = {"0.0": 0, "0.1": 1, "0.2": 2, "0.3": 3, "0.4": 4, "0.5": 5, "0.6": 6, "0.7": 7, "0.8": 8, "0.9": 9, "no_match": 0}
    transition_probability = tm.transition_matrix
    print("\nTransition probability:\n", transition_probability)
    state_probability = np.empty(len(states)); state_probability.fill(1/len(states))
    # model = hmm.CategoricalHMM(n_components=n_states)
    # model.startprob_ = state_probability
    # model.transmat_ = transition_probability
    model = hmm.CategoricalHMM(n_components=n_states)
    video_frame_process("videos/MSK_17.mp4", state_probability)
    # while True:
    #     zaal = input("Enter zaal: ")
    #     score = input("Enter score: ")
    #     calculate_hmm(score, zaal, chances_FP, chances_TP, state_probability)
        # print("detected zaal", detected_zaal)
    








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