from Evaluator import Evaluator
import glob
import cv2 as cv

from Extractor import Extractor
from Rectifier import Rectifier

# TODO

# extractor = Extractor()
# for img_path in glob.glob("./data/test_pictures_msk/*.jpg"):
#     image = cv.imread(img_path)
#     results = extractor.extract_and_crop_image(image, True)
#     cv.imshow("Original", cv.resize(image, (int(image.shape[1]/5),int(image.shape[0]/5))))
#     for idx, extracted_painting in enumerate(results):
#         cv.imshow(f"Painting {idx}", cv.resize(extracted_painting, (int(extracted_painting.shape[1]/5),int(extracted_painting.shape[0]/5))))
#     cv.waitKey(0)
#     cv.destroyAllWindows()



# evaluator = Evaluator()
# evaluator.evaluate_all_images("./data/Database2", "./data/Database_log.csv")


rectifier = Rectifier()
rectifier.calibrate_by_video("telin.ugent.be/~dvhamme/computervisie_2022/videos/gopro/calibration_W.mp4")


print("calibrated")
cap = cv.VideoCapture("telin.ugent.be/~dvhamme/computervisie_2022/videos/gopro/MSK_15.mp4")
fps = int(cap.get(cv.CAP_PROP_FPS))

f = 0
while cap.isOpened():
    process = False
    ret, frame = cap.read()
    if not ret:
        break
    if f < fps:
        f += 1
    if f % fps == 0:
        f = 1
        rectified_img = rectifier.process_image(frame)
        cv.imshow("Image", frame)
        cv.imshow("Rectified", rectified_img)
        cv.waitKey(0)
        cv.destroyAllWindows()