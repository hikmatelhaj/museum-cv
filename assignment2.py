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
        # range of cv.HISTCMP_BHATTACHARYYA is between 0 and 1
        scores += cv.compareHist(histogram1[i], histogram2[i] ,cv.HISTCMP_BHATTACHARYYA)
    return scores / len(histogram1)


def compare_hog_features(hog1, hog2):
    return euclidean(hog1, hog2)

def calc_histogram(img):
    hist_blue = cv.calcHist([img],[0],None,[256],[0,256])
    hist_green = cv.calcHist([img],[1],None,[256],[0,256])
    hist_red = cv.calcHist([img],[2],None,[256],[0,256])

    return hist_blue, hist_green, hist_red

def plot_histograms(histogram1, histogram2):
    plt.plot(histogram1, color='b')
    plt.show()
    plt.plot(histogram2, color='r')
    plt.show()
    plt.plot(histogram2, color='g')
    plt.show()


def get_final_score(scores):
    return np.average(scores, weights=[0.7, 0.3])


def affine_testje():
    import cv2
    import numpy as np

    # Load the input image
    img = cv2.imread(folder + "/" +files[0])

    rows, cols = img.shape[:2]

    # Define the vertices of the parallelogram
    pt1 = np.float32([0, 0])
    pt2 = np.float32([cols - 1, 0])
    pt3 = np.float32([cols*0.25, rows - 1])
    pt4 = np.float32([cols*0.75, rows - 1])

    # Compute the transformation matrix
    src_pts = np.array([pt1, pt2, pt3, pt4], dtype=np.float32)
    dst_pts = np.array([[0, 0], [cols - 1, 0], [0, rows - 1], [cols - 1, rows - 1]], dtype=np.float32)
    M = cv2.getPerspectiveTransform(src_pts, dst_pts)

    # Apply the transformation to the image
    result = cv2.warpPerspective(img, M, (cols, rows))
    result = cv2.rotate(result, cv2.ROTATE_90_CLOCKWISE)

    # Display the result
    cv2.imwrite('output_image.jpg', result)


if __name__ == "__main__":
    
    for file1 in files:
        all_scores_without_self = []
        score_with_self = 0
        for file2 in files:
            # print(folder + "/" + file1)
            img = cv.imread(folder + "/" + file1)
            img2 = cv.imread(folder + "/" + file2)
            # img2 = cv.imread('output_image.jpg')
            
        
            # Initiate ORB detector, argument for number of keypoints
            # orb = cv.ORB_create(500)
            orb = cv.ORB_create(500)
            
            # find the keypoints with ORB
            # print(img.shape)
            # print(file, files[1])
            # find the keypoints and descriptors with SIFT
            kp1, des1 = orb.detectAndCompute(img,None)
            kp2, des2 = orb.detectAndCompute(img2,None)

            # BFMatcher with default params
            bf = cv.BFMatcher()
            matches = bf.knnMatch(des1, des2, k=2)
            # matches = bf.match(des1,des2)
        
            # Apply ratio test
            """
            What is Lowe's ratio test? 

            Short version: each keypoint of the first image is matched with a number of keypoints from the second image. We keep the 2 best matches for each keypoint (best matches = the ones with the smallest distance measurement). Lowe's test checks that the two distances are sufficiently different. If they are not, then the keypoint is eliminated and will not be used for further calculations.
            """
            
            good = []
            lowe_ratio = 0.80
            for m,n in matches:
                if m.distance < lowe_ratio*n.distance:
                    good.append([m])
                    
            MIN_MATCH_COUNT = 10 # minimum number of matches required to estimate transformation matrix
            if len(good) >= MIN_MATCH_COUNT:
                src_pts = np.float32([kp1[m[0].queryIdx].pt for m in good]).reshape(-1, 1, 2)
                dst_pts = np.float32([kp2[m[0].trainIdx].pt for m in good]).reshape(-1, 1, 2)

                # Estimate transformation matrix using RANSAC
                M, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC, 5.0)
                
                # Count inliers
                inliers = np.sum(mask)

                # Compute ratio of inliers
                inlier_ratio = inliers / len(good)
            else:
                # Not enough matches were found, so the score is 0
                inlier_ratio = 0

            
            score_matcher = inlier_ratio
            hist_blue1, hist_green1, hist_red1 = calc_histogram(img)
            hist_blue2, hist_green2, hist_red2 = calc_histogram(img2)
            
            # plot_histograms([hist_blue, hist_green, hist_red], [hist_green, hist_blue, hist_red])

            histogram_score = compare_histograms([hist_blue1, hist_green1, hist_red1], [hist_blue2, hist_green2, hist_red2])
            final_score = get_final_score(np.array([score_matcher, 1 - histogram_score]))
            if file1 == file2:
                score_with_self = final_score
            else:
                all_scores_without_self.append(final_score)
            
            # ideeen: structuur foto --> hog, grootte kader
            # hog
            # fd, hog_image = hog(img, orientations=8, pixels_per_cell=(16, 16), cells_per_block=(1, 1), visualize=True, channel_axis=-1)
            # fd2, hog_image2 = hog(img2, orientations=8, pixels_per_cell=(16, 16), cells_per_block=(1, 1), visualize=True, channel_axis=-1)
            # # plt.axis("off")
            # # plt.imshow(hog_image, cmap="gray")
            # # plt.show()
            # print(fd.shape, fd2.shape)
            # print(compare_hog_features(fd, fd2))
            
            # Display nicely
            # img2 = cv.drawKeypoints(img2, kp2, None, color=(0,255,0), flags=0)
            # scale_percent = 30 # percent of original size
            # width = int(img2.shape[1] * scale_percent / 100)
            # height = int(img2.shape[0] * scale_percent / 100)
            # dim = (width, height)
            # resized = cv.resize(img2, dim, interpolation = cv.INTER_AREA)
            
            # img3 = cv.drawMatchesKnn(img,kp1,img2,kp2,good,None)
            # scale_percent = 30 # percent of original size
            # width = int(img3.shape[1] * scale_percent / 100)
            # height = int(img3.shape[0] * scale_percent / 100)
            # dim = (width, height)
            # resized = cv.resize(img3, dim, interpolation = cv.INTER_AREA)
            # # plt.imshow(img3),plt.show()
            # display("im", resized)
            break
            
        print("All the other scores are", all_scores_without_self)
        print("The average score is", np.mean(all_scores_without_self))
        print("The score with itself is", score_with_self)
        break
    
    
# other matcher:

# img3 = cv.drawMatchesKnn(img,kp1,img2,kp2,good,None)
# img3 = cv.drawMatches(img,kp1,img2,kp2,matches[:20],None)
# scale_percent = 30 # percent of original size
# width = int(img3.shape[1] * scale_percent / 100)
# height = int(img3.shape[0] * scale_percent / 100)
# dim = (width, height)
# resized = cv.resize(img3, dim, interpolation = cv.INTER_AREA)
# # plt.imshow(img3),plt.show()
# display("im", resized)

"""
Vragenlijst:
1. Wat is de beste metriek om matches te vergelijken? (# good matches en dan normalizen of met ransac  (tranform M en dan inliers) ?)
2. Hog, afbeelding rescalen, kunnen we dat doen?
3. Moet de paper in latex geschreven worden?
4. Misschien video sample rate vraag
"""