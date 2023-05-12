import ast
import glob
import pandas as pd
import numpy as np
import cv2 as cv
from Extractor import Extractor
from shapely.geometry import Polygon

class Evaluator:
    """
    Analytics of the different components
    """
    def __init__(self):
        self.extractor = Extractor()

    def calculate_iou(self, ground_truth_polygon, predicted_polygon):
        """
        Calculates the Intersection over Union of the groundtruth polygon and the predicted polygon
        :return: Error if polygons can be made / -1 if they don't intersect / iou value (value between 0 and 1)
        """
        gt_polygon = Polygon(ground_truth_polygon)
        pred_polygon = Polygon(predicted_polygon)

        if not gt_polygon.is_valid or not pred_polygon.is_valid:
            # one of the lists could not be converted to a polygon
            raise AssertionError("Invalid shape")
        
        if gt_polygon.intersects(pred_polygon):
            # the two polygons intersect
            intersection = gt_polygon.intersection(pred_polygon).area
            union = gt_polygon.union(pred_polygon).area

            return intersection / union
        
        # they don't intersect
        return -1

    def evaluate_single_polygon(self, image, ground_truth_polygon=[]) -> tuple[bool, int]:
        """
        Evaluate a single image polygon extraction. This will check if the `ground_truth_polygon` is inside the extracted_polygons
        :return: (is_found, iou)
        """
        predicted_polygons = self.extractor.find_painting(image)
        
        for predicted_polygon in predicted_polygons:
            try:
                iou = self.calculate_iou(ground_truth_polygon, predicted_polygon)

                if iou != -1:
                    # the polygon is not inside the extracted polygons
                    return (False, 0)
                
                return (True, iou)

            except AssertionError:
                # whoops, something went wrong
                continue

    def evaluate_image(self, image, ground_truth_polygons=[]):
        """
        This checks the painting extractions of a single image. 
        Compares the ground_truth polygons with the predicted polygons.
        :return: (# false negatives, # false positives, average iou)
        """
        predicted_polygons = self.extractor.find_painting(image)

        sum_iou = 0     # sum of the iou of the bounding boxes
        fn = 0          # paintings missed

        for gt_polygon in ground_truth_polygons:
            gt_found_in_pred = False    # is this painting extracted?
            max_iou = -1                # maximal IOU found for this painting

            for pred_polygon in predicted_polygons:
                try:
                    iou = self.calculate_iou(gt_polygon, pred_polygon)
                    if iou != -1:
                        # polygon is extracted
                        gt_found_in_pred = True
                        if iou>max_iou:
                            max_iou = iou

                except AssertionError:
                    continue
            
            if not gt_found_in_pred:
                # this painting is missed: false negative
                fn += 1
            else:
                # painting has a match: add best match iou
                sum_iou += max_iou
        
        # check if there are too many bounding boxes: probably non-paintings extracted
        fp = len(predicted_polygons)-len(ground_truth_polygons) if len(predicted_polygons)>len(ground_truth_polygons) else 0

        return (fn, fp, sum_iou/len(ground_truth_polygons))
            
    def evaluate_all_images(self, images_folder_path, log_path, verbose=0):
        """
        This checks the extractions of all images.
        Verbose: 0=no logs, 1=per image log line, 2=show images with polygons
        """
        log = pd.read_csv(log_path, skiprows=0)

        # convert string to list
        log["Top-left"] = log["Top-left"].apply(ast.literal_eval)
        log["Top-right"] = log["Top-right"].apply(ast.literal_eval)
        log["Bottom-left"] = log["Bottom-left"].apply(ast.literal_eval)
        log["Bottom-right"] = log["Bottom-right"].apply(ast.literal_eval)

        total_fp = 0
        total_fn = 0
        total_iou = 0
        total_imgs = 0

        for img_path in glob.glob(f"{images_folder_path}/*/*.jpg"):
            _, zaal, img_name = img_path.split("\\")
            img_name = img_name.strip(".jpg")

            img_paintings_gt = []
            img_log_entries = log[(log['Room'] == zaal) & (log['Photo'] == img_name)]

            for idx, img_log_entry in img_log_entries.iterrows():
                img_paintings_gt.append(
                    np.array([img_log_entry["Top-left"], img_log_entry["Top-right"],
                           img_log_entry["Bottom-right"], img_log_entry["Bottom-left"]]).reshape((-1, 2))
                )

            image = cv.imread(img_path)
            if len(img_paintings_gt) == 0:
                print(f"<!> {img_path} does not contain any painting")
                continue
            fn, fp, avg_iou = self.evaluate_image(image, img_paintings_gt)

            total_imgs += 1
            total_iou += avg_iou
            total_fn += fn
            total_fp += fp

            if verbose == 1:
                print(f"{img_path}:\t\t{fn} missed\t\t{fp} non-paintings extracted\t\t{avg_iou} IOU")
            if verbose == 2:
                image = cv.polylines(image, img_paintings_gt, True, (0, 255, 0), 10)
                cv.imshow("Original image", cv.resize(image, (int(image.shape[1]/5),int(image.shape[0]/5))))
                cv.waitKey(0)
                cv.destroyAllWindows()

        print(f"Report of all images ({total_imgs}):\n\tMissed paintings:\t {total_fn}\n\tExtracted non-paintings: {total_fp}\n\tAverage IoU:\t\t {total_iou/total_imgs}")
                

