# imports
import cv2 as cv
import numpy as np
import pandas as pd
import os
import glob
import ast
from shapely.geometry import Polygon
import os

# calculate intersection over union of bounding box
def bb_iou(gt_bb, pred_bb):
    gt_bb_shape = Polygon(gt_bb)
    pred_bb_shape = Polygon(pred_bb)

    intersection = gt_bb_shape.intersection(pred_bb_shape)
    if intersection:
        intersection_area = intersection.area
        union_area = gt_bb_shape.union(pred_bb_shape).area
        return intersection_area / union_area

    return -1


# extract bounding box van
def findPainting(image, returnBoundingBox=False):

    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    sigma = 0.40
    v = np.median(gray)
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    L2Gradient = True
    # ret, thresh = cv.threshold(blur, 1, 255, cv.THRESH_BINARY+cv.THRESH_OTSU)

    edges = cv.Canny(gray, 70, 250)
    dilated = cv.dilate(edges, np.ones((5, 5), np.uint8))

   # Get all contours of the edges
    contours, hierarchy = cv.findContours(
        dilated, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv.contourArea, reverse=True)[:10]

    # Check for every contour if it is a polygon with 4 corners
    approx_list = []
    for c in contours:
        app = cv.approxPolyDP(c, 0.05*cv.arcLength(c, True), True)
        if len(app) == 4:
            app = app.reshape(4, 2)
            approx_list.append(app)

    if returnBoundingBox:
        bb_list = []
        for approx in approx_list:
            x, y, w, h = cv.boundingRect(approx)
            bb_list.append(np.array([[x, y+h], [x+w, y+h], [x+w, y], [x, y]]))
        return bb_list
    else:
        return approx_list

def applyMeanShift(image):

    filtered_img = cv.pyrMeanShiftFiltering(image, 10, 20)

    return filtered_img
    
def getMask(img):
    hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    hsv = cv.resize(hsv, (int(hsv.shape[0]/5), int(hsv.shape[1]/5)))
    channels = cv.split(hsv)
    colors = ("h", "s", "v")
    lower = []
    upper = []
    for (channel, color) in zip(channels, colors):
        m = 10
        hist = cv.calcHist([channel], [0], None, [m], [0, 256])
        bin = hist.argmax()

        lower.append(((bin)/(m))*256)
        upper.append(((bin+1)/(m))*256)
        print(bin)

    print(lower)
    print(upper)

    mask = cv.inRange(hsv, (lower[0]-50, lower[1]-50, lower[2]-50), (upper[0]+50, upper[1]+50, upper[2]+50))
    kernel = np.ones((5, 5), np.uint8)
    mask = cv.dilate(mask, kernel=kernel, iterations=2)
    mask = cv.bitwise_not(mask)
    mask = cv.erode(mask, kernel, iterations=2)

    mask = cv.resize(mask, (int(img.shape[1]), int(img.shape[0])))
    print(mask.shape)
    print(img.shape)
    img_binary = cv.bitwise_and(img, img, mask=mask)
    return img_binary


def loop_paintings():
    img_path = "./data/Database2/Zaal_A/20190323_111313.jpg"
    img_path = "./data/Database2/Zaal_A/20190323_111327.jpg"

    imgs_path = "./data/Database"
    clear_path = "./data/Database2"
    log_path = "./data/Database_log.csv"

    log = pd.read_csv(log_path, skiprows=0)

    log["Top-left"] = log["Top-left"].apply(ast.literal_eval)
    log["Top-right"] = log["Top-right"].apply(ast.literal_eval)
    log["Bottom-left"] = log["Bottom-left"].apply(ast.literal_eval)
    log["Bottom-right"] = log["Bottom-right"].apply(ast.literal_eval)

    iou_sum = 0
    bb_count = 0

    for img_path in glob.glob(f"{clear_path}/*/*.jpg"):
        _, zaal, afb_naam = img_path_excl = img_path.split("\\")
        afb_naam = afb_naam.strip(".jpg")

        img_row = log[(log['Room'] == zaal) & (log['Photo'] == afb_naam)]

        orig_img = cv.imread(img_path)
        img = orig_img.copy()

        # img = cv.resize(img, (int(img.shape[0]/5), int(img.shape[1]/5)))

        # img = applyMeanShift(img)
        img = getMask(img)
        bb_list = findPainting(img)

        for _, el in img_row.iterrows():
            pts = np.array([el["Top-left"], el["Top-right"],
                           el["Bottom-right"], el["Bottom-left"]])
            pts = pts.reshape((-1, 2))
            img = cv.polylines(img, [pts], True, (0, 255, 0), 10)

            for bb in bb_list:
                img = cv.polylines(img, [bb], True, (0, 0, 255), 10)
                try:
                    iou = bb_iou(pts, bb)
                    if iou != -1:
                        iou_sum += iou
                        bb_count += 1

                except:
                    continue
        cv.imshow("X", cv.resize(img, (int(img.shape[0]/5), int(img.shape[1]/5))))
        cv.waitKey()
        cv.destroyAllWindows()

    print(f"Average I-o-U: {iou_sum/bb_count}")


loop_paintings()
