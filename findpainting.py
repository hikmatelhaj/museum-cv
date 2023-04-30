import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import sklearn
from sklearn.cluster import MeanShift, estimate_bandwidth


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
    channels = cv.split(img)
    colors = ("b", "g", "r")
    lower = []
    upper = []
    for (channel, color) in zip(channels, colors):
        hist = cv.calcHist([channel], [0], None, [256], [0, 256])
        y, x, p = plt.hist(hist)
        
        bin = y.argmin()

        lower.append(np.sum(y[:bin-1]))
        upper.append(np.sum(y[:bin]))
        print("---")

        
    # lower = [x - 50 for x in lower]
    # upper = [x + 50 for x in upper]
    print(lower)
    print(upper)
    mask = cv.inRange(img, (lower[0], lower[1], lower[2]), (upper[0], upper[1], upper[2]))

    img_binary = cv.bitwise_and(img, img, mask=mask)
    
    cv.imshow("mask", cv.resize(mask, dsize=(int(mask.shape[1]/5), int(mask.shape[0]/5))))
    # cv.imshow("mask", cv.resize(img, dsize=(int(img.shape[1]/5), int(img.shape[0]/5))))
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


def main():
    img_path = "./data/Database2/Zaal_A/20190323_111327.jpg"
    img = cv.imread(img_path)
    hsv =  cv.cvtColor(img, cv.COLOR_BGR2HSV)
    img = applyMeanShift(img)
    getMask(img)
    # applyMeanShift(img)
main()