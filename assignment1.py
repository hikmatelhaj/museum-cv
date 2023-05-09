# imports
import cv2 as cv
import numpy as np
import pandas as pd
import glob
import ast
from shapely.geometry import Polygon
import os

# calculate intersection over union of bounding box
def bb_iou(gt_bb, pred_bb):
    gt_bb_shape = Polygon(gt_bb)
    pred_bb_shape = Polygon(pred_bb)

    if not gt_bb_shape.is_valid or not pred_bb_shape.is_valid:
        return -1

    if gt_bb_shape.intersects(pred_bb_shape):
        intersection = gt_bb_shape.intersection(pred_bb_shape)
        intersection_area = intersection.area
        union_area = gt_bb_shape.union(pred_bb_shape).area

        return intersection_area / union_area
    
    return -1

# extract painting (polygon shape) out of image
def findPainting(image):
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    ret, th = cv.threshold(gray, 0, 255, cv.THRESH_BINARY+cv.THRESH_OTSU)
    blur = cv.GaussianBlur(th,(5,5),0)
    edges = cv.Canny(blur, 50, 120)
    dilated = cv.dilate(edges, np.ones((5, 5), np.uint8), iterations=3)
    
    contours = cv.findContours(dilated, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)[0]
    contours = sorted(contours, key=cv.contourArea, reverse=True)[:10]
    
    cut_list = []
    for c in contours:
        polygon = cv.approxPolyDP(c, 0.05*cv.arcLength(c, True), True)
        if len(polygon) == 4:
            polygon = polygon.reshape(4, 2)
            if Polygon(polygon).area > 15000:
                cut_list.append(polygon)
    return cut_list

# crop image based on polygon of painting
def transform(image, polygon):
    maxW = max(polygon[:,0]) - min(polygon[:,0])
    maxH = max(polygon[:,1]) - min(polygon[:,1])
    bounding_box = np.array([
        [0, 0],
        [0, maxH],
        [maxW, maxH],
        [maxW, 0]], dtype="float32")
    print(polygon)
    print(bounding_box)
    transform = cv.getPerspectiveTransform(np.float32(polygon), bounding_box)
    result = cv.warpPerspective(image, transform, (maxW, maxH))
    return result

# process single image and transform to cropped image
def process_single_image(image_path, drawPolygons=True):
    print(f"Processing {image_path}")
    img = cv.imread(image_path)
    polygons = findPainting(img)

    result_imgs = []

    for polygon in polygons:
        if drawPolygons:
            img = cv.polylines(img, [polygon], True, (0, 0, 255), 10)
        
        maxW = max(polygon[:,0]) - min(polygon[:,0])
        maxH = max(polygon[:,1]) - min(polygon[:,1])
        bounding_box = np.array([
            [0, 0],
            [0, maxH],
            [maxW, maxH],
            [maxW, 0]], dtype="float32")
        transform = cv.getPerspectiveTransform(np.float32(polygon), bounding_box)
        result = cv.warpPerspective(img, transform, (maxW, maxH))
        result_imgs.append(result)
    
    return result_imgs

# some statistics of a single image (intersection over union)
def analytics_single_image(image_path, ground_truth_polygon=[]):
    img = cv.imread(image_path)
    polygons = findPainting(img)

    for polygon in polygons:
        iou = bb_iou(ground_truth_polygon, polygon)
        if iou != -1:
            return iou
    return 1
    

# loop over all frames and save extracted paintings in `save_path` path
def loop(save_path="./extracted_paintings/", drawPolygons=False):
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    if not os.path.isdir(save_path):
        print(f"{save_path} is not a valid directory")
    frames_path = "./data/Database2"        # path of unprocessed images
    for img_path in glob.glob(f"{frames_path}/*/*.jpg"):
        file_name = img_path.split("\\")[-1]
        results = process_single_image(img_path, drawPolygons)
        for idx, extracted_painting in enumerate(results):
            ret = cv.imwrite(f"{save_path}/{file_name.strip('.jpg')}_{idx}.jpg", extracted_painting)
            if not ret: 
                print(f"Saving failed: {save_path}/{file_name.strip('.jpg')}_{idx}.jpg")

def loop_analytics():
    print(f"Processing analytics...")
    frames_path = "./data/Database2"        # path of unprocessed images
    log_path = "./data/Database_log.csv"    # path of database log

    log = pd.read_csv(log_path, skiprows=0)

    log["Top-left"] = log["Top-left"].apply(ast.literal_eval)
    log["Top-right"] = log["Top-right"].apply(ast.literal_eval)
    log["Bottom-left"] = log["Bottom-left"].apply(ast.literal_eval)
    log["Bottom-right"] = log["Bottom-right"].apply(ast.literal_eval)

    iou_sum = 0
    iou_len = 0

    for img_path in glob.glob(f"{frames_path}/*/*.jpg"):
        _, zaal, afb_naam = img_path.split("\\")
        afb_naam = afb_naam.strip(".jpg")

        img_row = log[(log['Room'] == zaal) & (log['Photo'] == afb_naam)]
        
        for _, el in img_row.iterrows():
            pts = np.array([el["Top-left"], el["Top-right"],
                           el["Bottom-right"], el["Bottom-left"]]).reshape((-1, 2))
            iou_avg_img = analytics_single_image(img_path, pts)
            iou_sum += iou_avg_img
            iou_len += 1
    
    print(f"Average Intersection-over-Union over all images:\n\t{iou_sum/iou_len}")

# extract all paintings out of all images:
loop()

# analytics of all images
# loop_analytics()