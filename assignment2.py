# https://docs.opencv.org/4.x/d1/d89/tutorial_py_orb.html

# https://stackoverflow.com/questions/39527947/how-to-calculate-score-from-orb-algorithm

import os
import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
from skimage.feature import hog
from scipy.spatial.distance import euclidean

folder = "Database_paintings/Database"
files = os.listdir(folder)

def display(title, img):
    cv.imshow(title, img)
    cv.waitKey(0); 
    cv.destroyAllWindows()


def compare_histograms(histogram1, histogram2):
    scores = 0
    for i in range(len(histogram1)):
        # 0.0 if the two histograms are identical
        scores += cv.compareHist(histogram1[i], histogram2[i] ,cv.HISTCMP_BHATTACHARYYA)
    return scores / len(histogram1)


def compare_hog_features(hog1, hog2):
    return euclidean(hog1, hog2)


def plot_histograms(histogram1, histogram2):
    plt.plot(histogram1, color='b')
    plt.show()
    plt.plot(histogram2, color='r')
    plt.show()
    plt.plot(histogram2, color='g')
    plt.show()




for file in files:
    img = cv.imread(folder + "/" + file)
    
    # Initiate ORB detector
    orb = cv.ORB_create()
    
    # find the keypoints with ORB
    print(img.shape)
    print(file)
    kp = orb.detect(img,None)
    
    
    hist_blue = cv.calcHist([img],[0],None,[256],[0,256])
    hist_green = cv.calcHist([img],[1],None,[256],[0,256])
    hist_red = cv.calcHist([img],[2],None,[256],[0,256])
    
    # plot_histograms([hist_blue, hist_green, hist_red], [hist_green, hist_blue, hist_red])

    score = compare_histograms([hist_blue, hist_green, hist_red], [hist_green, hist_blue, hist_red])
    print(score)
    
    # ideeen: structuur foto --> hog, grootte kader
    # hog
    fd, hog_image = hog(img, orientations=8, pixels_per_cell=(16, 16), cells_per_block=(1, 1), visualize=True, channel_axis=-1)
    plt.axis("off")
    plt.imshow(hog_image, cmap="gray")
    plt.show()
    print(fd)
        
    # Display nicely
    # img2 = cv.drawKeypoints(img, kp, None, color=(0,255,0), flags=0)
    # scale_percent = 30 # percent of original size
    # width = int(img2.shape[1] * scale_percent / 100)
    # height = int(img2.shape[0] * scale_percent / 100)
    # dim = (width, height)
    # resized = cv.resize(img2, dim, interpolation = cv.INTER_AREA)

    # display("im", resized)
    break