import cv2 as cv
import assignment1

def video_frame_process(video_path, frame_reduce=5):
    cap = cv.VideoCapture(video_path)
    f = 0
    while True:
        flag, frame = cap.read()
        if f % frame_reduce == 0:
            cv.imshow('video', frame)


            results = assignment1.process_single_image(frame)
            for idx, extracted_painting in enumerate(results):
                cv.imshow(f'painting {idx}', extracted_painting)

            cv.waitKey()
            for i in range(len(results)):
                cv.destroyWindow(f'painting {i}')
        
            if cv.waitKey(10) == 27:
                break
        f += 1

video_frame_process("./telin.ugent.be/~dvhamme/computervisie_2022/videos/smartphone/MSK_01.mp4")