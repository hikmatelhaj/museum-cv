from Evaluator import Evaluator
import glob
import cv2 as cv

from Extractor import Extractor

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