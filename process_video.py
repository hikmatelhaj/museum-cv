import cv2 
import numpy as np
import assignment1
from assignment2.assignment2 import *
import math
import json

def show_to_screen(img, title="image"):
    cv2.imshow(title, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# def video_frame_process(video_path):
#     cap = cv2.VideoCapture(video_path)
#     fps = cap.get(cv2.CAP_PROP_FPS)
#     frame_reduce = int(fps)
#     # import ipdb; ipdb.set_trace()
#     f = 0
#     frames = 0
#     bins_FP = {}
#     bins_TP = {}
#     bins_FP[0.0] = 0
#     bins_TP[0.0] = 0
#     while True:
#         flag, frame = cap.read()
#         if frame is None: # if video is over
#             break
#         gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#         lol = cv2.Laplacian(gray, cv2.CV_64F).var()
#         if f < 30:
#             f += 1
#         frames += 1
#         process = False
#         if lol > 30:
#             process = True
#             # Frame is not blurry
#         else:
#             continue

#         # wait until next sharp frame ( min 30 frames in between processing of sharp frames)
#         if process and f % frame_reduce == 0:
#             passed_seconds = int(frames / fps)
#             frames = 0
#             f = 0
#             process = False
#             print("processing frame")
            
#             for _ in range(passed_seconds):
#                 # bins_TP[0.0] += 1
#                 bins_FP[0.0] += 1
            
#             results = assignment1.process_single_image(frame)
#             for idx, extracted_painting in enumerate(results):
#                 # cv2.imwrite(f'./dataset_video/painting_{idx}', extracted_painting)
#                 scores, files = calculate_score_assignment2_multi(extracted_painting, "Database_paintings/Database")
#                 print("calculated")
#                 canvas = np.zeros((600, 1000, 3), dtype=np.uint8)
                
#                 scores = np.array(scores)
#                 ind = np.argpartition(scores, -2)[-2:]
#                 top2 = scores[ind]
#                 files = np.array(files)
#                 files = files[ind]

#                 score_bin = math.floor(top2[1] * 10) / 10
#                 if score_bin == 1.0:
#                     score_bin = 0.9


                
#                 img1 = cv2.imread("Database_paintings/Database/" + files[0]) # DB foto lowest
#                 img2 = cv2.imread("Database_paintings/Database/" + files[1]) # DB foto highest
#                 img3 = extracted_painting # extracted

#                 img1 = cv2.resize(img1, (250, 250))
#                 img2 = cv2.resize(img2, (250, 250)) 
#                 img3 = cv2.resize(img3, (250, 250))
                
#                 img3 = cv2.resize(img3, (350, 250))

#                 canvas = np.zeros((600, 800, 3), dtype=np.uint8)

#                 img1 = cv2.resize(img1, (250, 250))
#                 img2 = cv2.resize(img2, (250, 250))

#                 canvas[50:300, 50:300] = img1
#                 cv2.putText(canvas, f"DB lowest - {round(top2[0],2)}", (100, 350), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

#                 canvas[50:300, 350:600] = img2
#                 cv2.putText(canvas, f"DB highest {round(top2[1],2)}", (400, 350), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

#                 canvas[350:600, 225:575] = img3
#                 cv2.putText(canvas, f"Extracted", (275, 650), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

#                 cv2.imshow("Display Images", canvas)
#                 key = cv2.waitKey(0)
#                 # Check the pressed key and do something based on it
#                 if key == ord('y'):
#                     print("User pressed 'y'")
#                     if score_bin not in bins_TP.keys():
#                         bins_TP[score_bin] = 1
#                     else:
#                         bins_TP[score_bin] += 1

#                 elif key == ord('n'):
#                     print("User pressed 'n'")
#                     if score_bin not in bins_FP.keys():
#                         bins_FP[score_bin] = 1
#                     else:
#                         bins_FP[score_bin] += 1
                
#                 print("bins FP", bins_FP)
#                 print("bins TP", bins_TP)
#                 cv2.destroyAllWindows()
            

# video_frame_process("videos/MSK_03.mp4")



def video_frame_process(video_path):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_reduce = int(fps)
    # import ipdb; ipdb.set_trace()
    bins_FP = {}
    bins_TP = {}
    bins_FP[0.0] = 0
    bins_TP[0.0] = 0
    no_matched_found = 0
    f = 0
    caps = []
    lols = []
    filename_TP = f'labels/{os.path.basename(video_path)}_TP.json'
    filename_FP = f'labels/{os.path.basename(video_path)}_FP.json'
    filename_no_match = f'labels/{os.path.basename(video_path)}_no_match.txt'
    with open(filename_TP, 'w') as file:
        json.dump(bins_TP, file)
        
    with open(filename_FP, 'w') as file:
        json.dump(bins_FP, file)
    
    with open(filename_no_match, "w") as file:
        file.write(str(no_matched_found))
    
    while True:
        f += 1
        process = False
        if f%2 == 0:
            flag, frame = cap.read()
            caps.append(frame)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            lol = cv2.Laplacian(gray, cv2.CV_64F).var()
            lols.append(lol)
        else:
            continue
        
        if f == (frame_reduce * 10):
            print("Selecting the best frame ...")
            # get the best frame and process
            process = True
            lols = np.array(lols)
            # highest_lol_index = np.argmax(lols)
            # to_check = caps[highest_lol_index]
            # results = assignment1.process_single_image(to_check)
            indices = np.argsort(-lols)
            # Use the indices to sort the array
            painting_detected = False
            for index in indices:
                result = assignment1.process_single_image(caps[index], False)
                if len(result) > 0:
                    to_check = caps[index]
                    results = result
                    painting_detected = True
                    break
            if not painting_detected:
                no_matched_found += 1
                print("geen enkele schilderij herkend")
                
        if process:
            caps = []
            lols = []
            f = 0
            process = False
            print("processing frame")
            # for _ in range(passed_seconds):
            #     # bins_TP[0.0] += 1
            #     bins_FP[0.0] += 1
            results = assignment1.process_single_image(to_check, False)
            for idx, extracted_painting in enumerate(results):
                # cv2.imwrite(f'./dataset_video/painting_{idx}', extracted_painting)
                print("calculating")
                scores, files = calculate_score_assignment2_multi(extracted_painting, "Database_paintings/Database")
                print("calculated")
                canvas = np.zeros((600, 1000, 3), dtype=np.uint8)
            
                    
                scores = np.array(scores)
                ind = np.argpartition(scores, -2)[-2:]
                top2 = scores[ind]
                files = np.array(files)
                files = files[ind]
                print(ind)
                print(top2)
                score_bin = math.floor(top2[1] * 10) / 10
                if score_bin == 1.0:
                    score_bin = 0.9
                
                img1 = cv2.imread("Database_paintings/Database/" + files[0]) # DB foto lowest
                img2 = cv2.imread("Database_paintings/Database/" + files[1]) # DB foto highest
                img2 = cv2.GaussianBlur(img2, (5, 5), 0)
                img1 = cv2.GaussianBlur(img1, (5, 5), 0)
                img3 = extracted_painting # extracted

                img1 = cv2.resize(img1, (250, 250))
                img2 = cv2.resize(img2, (250, 250)) 
                img3 = cv2.resize(img3, (250, 250))
                
                # img3 = cv2.resize(img3, (350, 250))

                canvas = np.zeros((600, 800, 3), dtype=np.uint8)

                img1 = cv2.resize(img1, (250, 250))
                img2 = cv2.resize(img2, (250, 250))

                canvas[50:300, 50:300] = img3
                cv2.putText(canvas, f"Extracted", (275, 650), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

                canvas[50:300, 350:600] = img2
                cv2.putText(canvas, f"DB highest {round(top2[1],2)}", (400, 350), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

                to_check = cv2.resize(to_check, (350, 250)) #  into shape (250,350,3)
                canvas[350:600, 225:575] = to_check

                cv2.imshow("Display Images", canvas)
                key = cv2.waitKey(0)
                # Check the pressed key and do something based on it
                
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
            

video_frame_process("videos/MSK_03.mp4")