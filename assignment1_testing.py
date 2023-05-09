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

        # print(f"{intersection_area}\t\t{union_area} => {(intersection_area / union_area)*100}")
        return intersection_area / union_area

    # areaPrediction = 0
    # areaGroundTruth = 0
    # areaIntersection = 0
    # for x in range(0, 1000):
    #         for y in (range(0, 1000)):
    #             pointIsInPrediction = cv.pointPolygonTest(contour=pred_bb, pt=(x, y), measureDist=False) > 0
    #             pointIsInGroundTruth = cv.pointPolygonTest(contour=gt_bb, pt=(x, y), measureDist=False) > 0

    #             areaPrediction += pointIsInPrediction is True
    #             areaGroundTruth += pointIsInGroundTruth is True
    #             areaIntersection += (pointIsInPrediction is True and pointIsInGroundTruth is True)
    # if(areaIntersection == areaGroundTruth and areaPrediction > areaGroundTruth):
    #     areaGroundTruth, areaPrediction = areaPrediction, areaGroundTruth
            
    #                             # swap areas when green polygon encloses red polygon
                
    #     return areaIntersection / areaGroundTruth

    return -1


# extract bounding box van
def findPainting(image, returnBoundingBox=False, showImages=False):
    #adjust brightness and contrast
    # image = cv.convertScaleAbs(image, alpha=2, beta=0)
    image = cv.resize(image, (int(image.shape[1]/5), int(image.shape[0]/5)))
    cv.imshow("image", image)
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    # sigma = 0.40
    # v = np.median(gray)
    # lower = int(max(0, (1.0 - sigma) * v))
    # upper = int(min(255, (1.0 + sigma) * v))
    # L2Gradient = True
    # ret, thresh = cv.threshold(blur, 1, 255, cv.THRESH_BINARY+cv.THRESH_OTSU)
    ret, th = cv.threshold(gray, 0, 255, cv.THRESH_BINARY+cv.THRESH_OTSU)
    blur = cv.GaussianBlur(th,(5,5),0)
    edges = cv.Canny(blur, 50, 120)
    dilated = cv.dilate(edges, np.ones((5, 5), np.uint8), iterations=3)
    # eroded = cv.erode(dilated, np.ones((9, 9), np.uint8))
    # dilated = cv.dilate(eroded, np.ones((5, 5), np.uint8), iterations=1)
    
    # cv.imshow("eroded", cv.resize(eroded, (int(eroded.shape[0]/5), int(eroded.shape[1]/5))))

   # Get all contours of the edges
    contours = cv.findContours(dilated, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)[0]
    contours = sorted(contours, key=cv.contourArea, reverse=True)[:10]
    
    if showImages:
        cv.imshow("threshold", cv.resize(th, (int(th.shape[0]), int(th.shape[1]))))
        cv.imshow("blur", cv.resize(blur, (int(blur.shape[0]), int(blur.shape[1]))))
        cv.imshow("edge", cv.resize(edges, (int(edges.shape[0]), int(edges.shape[1]))))
        cv.imshow("dilated", cv.resize(dilated, (int(dilated.shape[0]), int(dilated.shape[1]))))
        # cv.imshow("eroded", cv.resize(eroded, (int(eroded.shape[0]/5), int(eroded.shape[1]/5))))
        cv.drawContours(image, contours, -1, (0,255,0), 3)
        cv.imshow("contours", cv.resize(image, (int(image.shape[0]), int(image.shape[1]))))

    # Check for every contour if it is a polygon with 4 corners
    approx_list = []
    for c in contours:
        app = cv.approxPolyDP(c, 0.05*cv.arcLength(c, True), True)
        if (len(app) == 4):
            app = app.reshape(4, 2)
            if (Polygon(app).area > 1000) :
                print("app", app)
                approx_list.append(app)

    if returnBoundingBox:
        bb_list = []
        for approx in approx_list:
            x, y, w, h = cv.boundingRect(approx)
            bb_list.append(np.array([[x*5, (y+h)*5], [(x+w)*5, (y+h)*5], [(x+w)*5, y*5], [x*5, y*5]]))
        return bb_list
    else:
        print(approx_list)
        approx_list = [i * 5 for i in approx_list]
        print('--')
        print(approx_list)
        print('------------')
        return approx_list

def applyMeanShift(image):

    filtered_img = cv.pyrMeanShiftFiltering(image, 10, 20)

    return filtered_img


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
        # img = getMask(img)
        bb_list = findPainting(img, showImages=False)

        # os.system('cls')
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
                        maxW = max(bb[:,0]) - min(bb[:,0])
                        maxH = max(bb[:,1]) - min(bb[:,1])
                        bounding_box = np.array([
                            [0, 0],
                            [0, maxH],
                            [maxW, maxH],
                            [maxW, 0]], dtype="float32")
                        print(bb)
                        print(bounding_box)
                        transform = cv.getPerspectiveTransform(np.float32(bb), bounding_box)
                        result = cv.warpPerspective(img, transform, (maxW, maxH))
                        cv.imshow(f"result {bb_count}", cv.resize(result, (int(result.shape[1]/3), int(result.shape[0]/3))))

                except Exception as e:
                    print("fout", e)
                    continue
        cv.imshow(afb_naam, cv.resize(img, (int(img.shape[0]/5), int(img.shape[1]/5))))
        cv.waitKey()
        cv.destroyAllWindows()

    print(f"Average I-o-U: {iou_sum/bb_count}")


loop_paintings()



# # def getMask(img):
#     hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
#     # hsv = cv.resize(hsv, (int(hsv.shape[0]/5), int(hsv.shape[1]/5)))
#     channels = cv.split(hsv)
#     colors = ("h", "s", "v")
#     lower = []
#     upper = []
#     for (channel, color) in zip(channels, colors):
#         m = 10
#         hist = cv.calcHist([channel], [0], None, [m], [0, 256])
#         bin = hist.argmax()

#         lower.append(((bin)/(m))*256)
#         upper.append(((bin+1)/(m))*256)
#         # print(bin)

#     # print(lower)
#     # print(upper)

#     mask = cv.inRange(
#         hsv, (lower[0], lower[1]-30, lower[2]-30), (upper[0], upper[1]+30, upper[2]+30))
#     kernel = np.ones((5, 5), np.uint8)
#     mask = cv.dilate(mask, kernel=kernel, iterations=2)
#     mask = cv.bitwise_not(mask)
#     mask = cv.erode(mask, kernel, iterations=2)

#     mask = cv.resize(mask, (int(img.shape[1]), int(img.shape[0])))

#     img[mask == 0] = 0
#     blurred_image = cv.blur(img, (5, 5))
#     blurred_mask = cv.blur(mask, (5, 5))
#     # result = blurred_image / blurred_mask
#     # print(mask.shape)
#     # print(img.shape)
#     # img_binary = cv.bitwise_and(img, img, mask=mask)
#     return blurred_image
