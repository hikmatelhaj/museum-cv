import cv2 
import numpy as np
from Extractor import *
from Rectifier import Rectifier
from assignment2.assignment2 import *
import math
import json

def show_to_screen(img, title="image"):
    cv2.imshow(title, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    

def make_directories(path):
    try: 
        os.mkdir(path) 
    except OSError as error: 
        pass
    
def video_frame_process(video_path, gopro=False, type="calibration_W"):
    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    if gopro:
        rectifier = Rectifier()
        rectifier.load_calibration(type)
        print("calibrated")

    f = 0
    no_matched_found = 0
    total_seconds = 0 # keep track of the total frames that was processed (can be converted to secs by dividing by fps)
    last_decile = 0 # keep track of the last 10 seconds that was processed
    
    bins_FP = {}
    bins_TP = {}
    bins_FP[0.0] = 0
    bins_TP[0.0] = 0
    bins_FP[0.1] = 0
    bins_TP[0.1] = 0
    bins_FP[0.2] = 0
    bins_TP[0.2] = 0
    bins_FP[0.3] = 0
    bins_TP[0.3] = 0
    bins_FP[0.4] = 0
    bins_TP[0.4] = 0
    bins_FP[0.5] = 0
    bins_TP[0.5] = 0
    bins_FP[0.6] = 0
    bins_TP[0.6] = 0
    bins_FP[0.7] = 0
    bins_TP[0.7] = 0
    bins_FP[0.8] = 0
    bins_TP[0.8] = 0
    bins_FP[0.9] = 0
    bins_TP[0.9] = 0
    scores_per_decile = {}
    st = time.time()

    extractor = Extractor()
    
    make_directories(f'labels/{os.path.basename(video_path)}')
    filename_TP = f'labels/{os.path.basename(video_path)}/{os.path.basename(video_path)}_TP.json'
    filename_FP = f'labels/{os.path.basename(video_path)}/{os.path.basename(video_path)}_FP.json'
    filename_no_match = f'labels/{os.path.basename(video_path)}/{os.path.basename(video_path)}_no_match.txt'
    with open(filename_TP, 'w') as file:
        json.dump(bins_TP, file)
        
    with open(filename_FP, 'w') as file:
        json.dump(bins_FP, file)
    
    with open(filename_no_match, "w") as file:
        file.write(str(no_matched_found))

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
            if gopro:
                frame = rectifier.process_image(frame)
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
            results = extractor.extract_and_crop_image(frame, False)
            for idx, extracted_painting in enumerate(results):
                if idx > 2: # process max 2 paintings per frame
                    continue
                # cv2.imwrite(f'./dataset_video/painting_{idx}', extracted_painting)
                scores, files = calculate_score_assignment2_multi(extracted_painting, "Database_paintings/Database")
                
                canvas = np.zeros((600, 1000, 3), dtype=np.uint8)
                # cv2.imshow("extracted_painting", extracted_painting)
                # cv2.imshow("frame", frame)
                # cv2.waitKey(0)
                # cv2.destroyAllWindows()
                
                scores = np.array(scores)
                ind = np.argpartition(scores, -2)[-2:]
                top2 = scores[ind]
                files = np.array(files)
                files = files[ind]
                print("calculated", top2[1])
                scores_per_decile[decile].append({"score": top2[1], "file": files[1], "extracted": extracted_painting, "to_check": frame})
                if last_decile == decile: # only if a new decile is reached, show the best results
                    continue
                
                items = scores_per_decile.get(last_decile, [])
                et = time.time()
                elapsed_time = et - st
                print('It took', elapsed_time, 'seconds to process the last', decile - last_decile ,'seconds')
                
                    
                if len(items) == 0:
                    amount_of_deciles_without_paintings = ((decile - last_decile) // 10)
                    no_matched_found += amount_of_deciles_without_paintings
                    with open(filename_no_match, "w") as file:
                        file.write(str(no_matched_found))
                    print("No paintings detected in the last", decile - last_decile , "seconds")
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
                if key == ord('y'):
                    print("User pressed 'y'")
                    if score_bin not in bins_TP.keys():
                        bins_TP[score_bin] = 1
                    else:
                        bins_TP[score_bin] += 1

                elif key == ord('n'):
                    print("User pressed 'n'")
                    if score_bin not in bins_FP.keys():
                        bins_FP[score_bin] = 1
                    else:
                        bins_FP[score_bin] += 1
                
                print("bins FP", bins_FP)
                print("bins TP", bins_TP)
                with open(filename_TP, 'w') as file:
                    json.dump(bins_TP, file)
                    
                with open(filename_FP, 'w') as file:
                    json.dump(bins_FP, file)
                
                with open(filename_no_match, "w") as file:
                    file.write(str(no_matched_found))
                    
                cv2.destroyAllWindows()
                st = time.time()
            

video_frame_process("videos/MSK_11.mp4", False, "calibration_W")
