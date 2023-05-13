import cv2 
import numpy as np
import assignment1
from assignment2.assignment2 import *


def show_to_screen(img, title="image"):
    cv2.imshow(title, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def video_frame_process(video_path):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_reduce = int(fps)
    # import ipdb; ipdb.set_trace()
    f = 0
    while True:

        flag, frame = cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        lol = cv2.Laplacian(gray, cv2.CV_64F).var()
        if f < 30:
            f += 1
        process = False
        if lol > 100:
            process = True
            # Frame is not blurry
        else:
            continue

        # wait until next sharp frame ( min 30 frames in between processing of sharp frames)
        if process and f % frame_reduce == 0:
            f = 0
            print("processing frame")
            results = assignment1.process_single_image(frame)
            for idx, extracted_painting in enumerate(results):
                # cv2.imwrite(f'./dataset_video/painting_{idx}', extracted_painting)
                scores, files = calculate_score_assignment2(extracted_painting, "Database_paintings/Database")
                print("calculated")
                canvas = np.zeros((600, 1000, 3), dtype=np.uint8)
                
                scores = np.array(scores)
                ind = np.argpartition(scores, -2)[-2:]
                top2 = scores[ind]
                files = np.array(files)
                files = files[ind]
                print(ind)
                print(top2)
                
                img1 = cv2.imread("Database_paintings/Database/" + files[0]) # DB foto lowest
                img2 = cv2.imread("Database_paintings/Database/" +files[1]) # DB foto highest
                img3 = extracted_painting # extracted

                img1 = cv2.resize(img1, (250, 250))
                img2 = cv2.resize(img2, (250, 250)) 
                img3 = cv2.resize(img3, (250, 250))
                
                img3 = cv2.resize(img3, (350, 250))

                canvas = np.zeros((600, 800, 3), dtype=np.uint8)

                img1 = cv2.resize(img1, (250, 250))
                img2 = cv2.resize(img2, (250, 250))

                canvas[50:300, 50:300] = img1
                cv2.putText(canvas, f"DB lowest - {round(top2[0],2)}", (100, 350), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

                canvas[50:300, 350:600] = img2
                cv2.putText(canvas, f"DB highest {round(top2[1],2)}", (400, 350), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

                canvas[350:600, 225:575] = img3
                cv2.putText(canvas, f"Extracted", (275, 650), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

                cv2.imshow("Display Images", canvas)
                key = cv2.waitKey(0)
                # Check the pressed key and do something based on it
                if key == ord('y'):
                    print("User pressed 'y'")
                elif key == ord('n'):
                    print("User pressed 'n'")
                    
                cv2.destroyAllWindows()
            

video_frame_process("videos/MSK_03.mp4")