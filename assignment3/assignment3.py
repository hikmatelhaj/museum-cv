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
from assignment2.assignment2 import *
from geomap import showHeatmap, heatmapToImg
from assignment1.Extractor import *
from assignment1.Rectifier import Rectifier
screen_scale_factor = 1.5

# Possible rooms
states = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12", "13", "14", "15", "16", "17", "18", "19", "RI", "II", "V"]

def print_green(text):
    print("\033[32m" + text + "\033[0m")


def process_video(video_path, state_probability, gopro=False, type="calibration_W"):
    """
    video_path: path to the video
    state_probability: the initial probability
    gopro: if the video is a gopro video
    type: the type of calibration
    """
    if gopro:
        rectifier = Rectifier()
        rectifier.load_calibration(type)
        print("calibrated")
    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    last_probabilities = state_probability
    last_last_probabilities = state_probability
    f = 0
    total_frames = 0 # keep track of the total frames that was processed (can be converted to secs by dividing by fps)
    last_decile = 0 # keep track of the last 10 seconds that was processed
    scores_per_decile = {}
    st = time.time()
    found_observations = [] # All observations in HMM
    
    extractor = Extractor()
    
    seconds_to_wait = 1 # wait x seconds before processing the next frame after a sharp frame
    process = False
    while True:
        
        total_frames += 1
        # if total_frames % 2 != 0:
        #     continue
        
        flag, frame = cap.read()
        
        if frame is None: # if video is over
            break
        if gopro:
            frame = rectifier.process_image(frame)
            
        if f < seconds_to_wait * fps: # logic to not process every frame
            f += 1
        if f % (seconds_to_wait * fps) == 0: # logic to not process every frame
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            lol = cv2.Laplacian(gray, cv2.CV_64F).var()
            if lol > 30: # A sharp frame is found
                process = True
                # Frame is not blurry
            else:
                continue

        # wait until next sharp frame (min 30 frames in between processing of sharp frames)
        if process and f % (seconds_to_wait * fps) == 0:
            f = 1
            process = False
            second = total_frames / fps # get the current second in the video
            decile = math.floor(second / 10) * 10 # round to the nearest decile (10, 20, 30, ...)
            if decile not in scores_per_decile.keys():
                scores_per_decile[decile] = []
            polygons, results, img_with_polygons = extractor.extract_and_crop_image(frame, True) # extract paintings from frame
            for idx, extracted_painting in enumerate(results):
                if idx > 2: # process max 2 paintings per frame for performance reasons
                    continue
                scores, files = calculate_score_assignment2_multi(extracted_painting, "data/database")
                
                canvas = np.zeros((int(600*screen_scale_factor), int(1000*screen_scale_factor), 3), dtype=np.uint8)
                scores = np.array(scores)
                files = np.array(files)
                
                score = np.max(scores)
                index_file = np.argmax(scores)
                file = files[index_file]
                if last_decile == decile: # only print the resultst of the current decile
                    print("Score", round(score, 2), "in room", get_zaal_by_filename(file))
                    scores_per_decile[decile].append({"score": score, "file": file, "extracted": extracted_painting, "to_check": frame, "frame_with_contours": img_with_polygons, "polygon": polygons[idx]})
                    continue # only if a new decile is reached, show the best results, else skip
                    
                
                et = time.time()
                elapsed_time = et - st
                print_green(f"It took {elapsed_time} seconds to process the last {decile - last_decile} seconds")
                # if decile - last_decile > 10:
                #     print(len(scores_per_decile.get(last_decile, [])), scores_per_decile.get(last_decile, []))
                #     print(len(scores_per_decile.get(last_decile - 10, [])), scores_per_decile.get(last_decile - 10, []))

                all_items = []
                for i in range(last_decile, decile, 10):
                    items = scores_per_decile.get(i, [])
                    all_items.append(items)
                
                for items in all_items:
                    if len(items) == 0:
                        # hmm calculate
                        found_observations.append("no_match")
                        try:
                            prev = found_observations[-2]
                        except:
                            prev = found_observations[-1]
                        
                        last_last_probabilities = last_probabilities
                        last_probabilities, zaal_predict, df, percentage = calculate_hmm("no_match", "no_match", chances_FP, chances_TP, last_last_probabilities, prev)
                        
                        if len(items) == 1:
                            print_green(f"No paintings detected in the last {decile - last_decile} seconds")
                        else:
                            print_green(f"No paintings detected")
                            
                        print_green(f"Room {zaal_predict} is predicted with {round(percentage*100, 2)}% certainty. There is no matching database image.")
                        # showHeatmap(df)
                        last_decile = decile # update the last decile
                        st = time.time()
                        
                        continue
                    
                    print_green(f"Currently in {decile} seconds")

                    # Find the item with the highest score
                    highest_score_item = max(items, key=lambda x: x["score"])

                    # Retrieve the desired information from the highest score item
                    highest_score = highest_score_item["score"]
                    file_name = highest_score_item["file"]
                    highest_extracted_painting = highest_score_item["extracted"]
                    highest_to_check = highest_score_item["to_check"]
                    highest_frame_with_contours = highest_score_item["frame_with_contours"]

                    highest_frame_with_contours = cv.polylines(highest_frame_with_contours, [highest_score_item["polygon"]], True, (0, 255, 0), 10)
                    
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
                    last_last_probabilities = last_probabilities
                    last_probabilities, zaal_predict, df, percentage = calculate_hmm(score_bin, zaal, chances_FP, chances_TP, last_last_probabilities, prev)
                    
                    print_green(f"Room {zaal_predict} is predicted with {round(percentage*100, 2)}% certainty. The matching database room is {zaal}.")
                    

                    img2 = cv2.imread("data/database/" + file_name) # highest match
                    img2 = cv2.GaussianBlur(img2, (5, 5), 0) # for displaying purposes
                    img3 = highest_extracted_painting # extracted

                    img2 = cv2.resize(img2, (int(250*screen_scale_factor), int(250*screen_scale_factor))) 
                    img3 = cv2.resize(img3, (int(250*screen_scale_factor), int(250*screen_scale_factor)))
                    highest_frame_with_contours = cv2.resize(highest_frame_with_contours, (int(350*screen_scale_factor), int(250*screen_scale_factor)))
                    
                    canvas = np.zeros((int(600*screen_scale_factor), int(1230*screen_scale_factor), 3), dtype=np.uint8)

                    canvas[int(25*screen_scale_factor):int(275*screen_scale_factor), int(50*screen_scale_factor):int(400*screen_scale_factor)] = highest_frame_with_contours
                    canvas[int(325*screen_scale_factor):int(575*screen_scale_factor), int(50*screen_scale_factor):int(300*screen_scale_factor)] = img2

                    cv2.putText(canvas, f"Match score: {round(highest_score, 2)}", (int(325*screen_scale_factor), int(535*screen_scale_factor)), cv2.FONT_HERSHEY_SIMPLEX, 1.8, (255, 255, 255), 2)
                    cv2.putText(canvas, f"Hall: {get_zaal_by_filename(file_name)}", (int(325*screen_scale_factor), int(570*screen_scale_factor)), cv2.FONT_HERSHEY_SIMPLEX, 1.8, (255, 255, 255), 1)

                    heatmapImg, colorbar = heatmapToImg(df)
                    heatmapImg = cv2.resize(heatmapImg, (int(600*screen_scale_factor), int(450*screen_scale_factor)))
                    colorbar = cv2.resize(colorbar, (int(30*screen_scale_factor), int(450*screen_scale_factor)))
                    canvas[int(25*screen_scale_factor):int(475*screen_scale_factor), int(540*screen_scale_factor):int(1140*screen_scale_factor)] = heatmapImg
                    canvas[int(25*screen_scale_factor):int(475*screen_scale_factor), int(1150*screen_scale_factor):int(1180*screen_scale_factor)] = colorbar

                    
                    cv2.putText(canvas, "1", (1185, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 1)
                    cv2.putText(canvas, "0", (1185, 475), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 1)
                    cv2.putText(canvas, f"Current hall: {zaal_predict}", (int(900*screen_scale_factor), int(535*screen_scale_factor)), cv2.FONT_HERSHEY_SIMPLEX, 1.8, (255, 255, 255), 2)
                    cv2.putText(canvas, f"Probability: {round(percentage, 2)}", (int(900*screen_scale_factor), int(570*screen_scale_factor)), cv2.FONT_HERSHEY_SIMPLEX, 1.8, (255, 255, 255), 1)

                    cv2.imshow("Display Images", canvas)
                    key = cv2.waitKey(0)
                    # Check the pressed key and do something based on it
                    # while key != ord('y') and key != ord('n'):
                    #     key = cv2.waitKey(0)
                    cv2.destroyAllWindows()
                    
                # clear to save space
                scores_per_decile[decile] = []
                # We add again the latest observation, because that one is not processed yet
                scores_per_decile[decile].append({"score": score, "file": file, "extracted": extracted_painting, "to_check": frame, "frame_with_contours": img_with_polygons, "polygon": polygons[idx]})
                print("Score", round(score, 2), "in room", get_zaal_by_filename(file))
                st = time.time()
            
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
    
    print(chances_TP)
    print(chances_FP)

    

    # Define the state space
    n_states = len(states)
    translation_event = {"0.0": 0, "0.1": 1, "0.2": 2, "0.3": 3, "0.4": 4, "0.5": 5, "0.6": 6, "0.7": 7, "0.8": 8, "0.9": 9, "no_match": 0}
    transition_probability = tm.transition_matrix
    print("Transition matrix:\n", transition_probability)
    
    # Define the initial state distribution: every state is equally likely
    state_probability = np.empty(len(states)); state_probability.fill(1/len(states))
    process_video("videos/MSK_07.mp4", state_probability, False, "calibration_W") # demo 1
    # process_video("videos/MSK_15.mp4", state_probability, True, "calibration_W") # demo 2