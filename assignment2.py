# https://docs.opencv.org/4.x/d1/d89/tutorial_py_orb.html

# https://stackoverflow.com/questions/39527947/how-to-calculate-score-from-orb-algorithm

import os
import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

folder = "Database_paintings/Database"
files = os.listdir(folder)

def display(title, img):
    cv.imshow(title, img)
    cv.waitKey(0)
    cv.destroyAllWindows()

for file in files:
    img = cv.imread(folder + "/" + file)
    
    # Initiate ORB detector
    orb = cv.ORB_create()
    
    # find the keypoints with ORB
    print(img.shape)
    print(file)
    kp = orb.detect(img,None)
    
    # ideeen: structuur foto --> hog, grootte kader
    
    
    hist_blue = cv.calcHist([img],[0],None,[256],[0,256])
    hist_green = cv.calcHist([img],[1],None,[256],[0,256])
    hist_red = cv.calcHist([img],[2],None,[256],[0,256])
    
    
    feature_vector_colors = np.array([hist_blue, hist_green, hist_red])

    plt.plot(hist_blue, color='b')
    plt.show()
    plt.plot(hist_red, color='r')
    plt.show()
    plt.plot(hist_green, color='g')
    plt.show()
    
    # Display nicely
    # img2 = cv.drawKeypoints(img, kp, None, color=(0,255,0), flags=0)
    # scale_percent = 30 # percent of original size
    # width = int(img2.shape[1] * scale_percent / 100)
    # height = int(img2.shape[0] * scale_percent / 100)
    # dim = (width, height)
    # resized = cv.resize(img2, dim, interpolation = cv.INTER_AREA)

    # display("im", resized)
    break