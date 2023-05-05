import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import colorsys

# def findPainting(image):
# #     Get edges and make them wider

#     hsv =  cv.cvtColor(image, cv.COLOR_BGR2HSV)
#     im_meanshifted = applyMeanShift(hsv)
#     lower = 0
#     upper = 0
#     img_masked = getMask(im_meanshifted, lower, upper)
#     #gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

#     cv.imshow("X",cv.resize(gray, dsize=(int(gray.shape[0]/5), int(gray.shape[1]/5))))
#     sigma = 0.40
#     v = np.median(gray)
#     lower = int(max(0, (1.0 - sigma) * v))
#     upper = int(min(255, (1.0 + sigma) * v))
#     L2Gradient = True
#     # ret, thresh = cv.threshold(blur, 1, 255, cv.THRESH_BINARY+cv.THRESH_OTSU)
    
#     edges = cv.Canny(gray, 70, 250)
#     dilated = cv.dilate(edges, np.ones((5,5), np.uint8))

#     cv.imshow("Y",cv.resize(dilated, dsize=(int(dilated.shape[0]/5), int(dilated.shape[1]/5))))

def getMask(img):
    hsv =  cv.cvtColor(img, cv.COLOR_BGR2HSV)
    channels = cv.split(img)
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
    # lower = [x - 50 for x in lower]
    # upper = [x + 50 for x in upper]
    
    # lower = np.array(colorsys.rgb_to_hsv(lower[2]/255, lower[1]/255, lower[0]/255))
    # upper = np.array(colorsys.rgb_to_hsv(upper[2]/255, upper[1]/255, upper[0]/255))

    print(lower)
    print(upper)

    # h,w,_ = img.shape
    # mask = np.zeros((h+2,w+2),np.uint8)
    mask = cv.inRange(img, (lower[0]-5, lower[1]-30, lower[2]-30), (upper[0]+5, upper[1]+30, upper[2]+30))
    kernel = np.ones((5, 5), np.uint8)
    mask = cv.dilate(mask, kernel=kernel, iterations=2)
    mask = cv.bitwise_not(mask)
    img2 = cv.cvtColor(img, cv.COLOR_HSV2BGR)

    img_binary = cv.bitwise_and(img2, img2, mask=mask)
    
    # cv.imshow("mask", cv.resize(mask, dsize=(int(mask.shape[1]/5), int(mask.shape[0]/5))))
    cv.imshow("mask", cv.resize(img_binary, dsize=(int(img_binary.shape[1]/5), int(img_binary.shape[0]/5))))
    cv.waitKey()
    cv.destroyAllWindows()

def applyMeanShift(image):
    # Apply mean shift filtering with a spatial window radius of 10 and a color window radius of 20
    #img = cv.resize(image, (int(image.shape[0]/5), int(image.shape[1]/5)))
    filtered_img = cv.pyrMeanShiftFiltering(image, 10, 20)

    return filtered_img
    # Display the input and filtered images
    #cv.imshow('Input Image', img - filtered_img)
    #cv.imshow('Filtered Image', filtered_img)    
    #cv.waitKey(0)
    #cv.destroyAllWindows()

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

def main():
    img_path = "./data/Database2/Zaal_1/IMG_20190323_111717.jpg"
    img = cv.imread(img_path)
    # img = applyMeanShift(img)
    masked = getMask(img)
    findPainting(masked)
    # applyMeanShift(img)
main()