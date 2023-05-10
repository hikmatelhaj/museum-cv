# https://docs.opencv.org/4.x/d1/d89/tutorial_py_orb.html

# https://stackoverflow.com/questions/39527947/how-to-calculate-score-from-orb-algorithm

import os
import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
from skimage.feature import hog
from scipy.spatial.distance import euclidean
import time


def display(title, img):
    cv.imshow(title, img)
    cv.waitKey(0)
    cv.destroyAllWindows()


def compare_histograms(histogram1: list[np.ndarray], histogram2: list[np.ndarray]) -> float:
    """
    Compare histograms using Bhattacharyya distance and return the average score.
    """
    scores = 0
    for i in range(len(histogram1)):
        # 0.0 if the two histograms are identical
        # range of cv.HISTCMP_BHATTACHARYYA is between 0 and 1
        scores += cv.compareHist(histogram1[i],
                                 histogram2[i], cv.HISTCMP_BHATTACHARYYA)
    return scores / len(histogram1)


def compare_hog_features(hog1: np.ndarray, hog2: np.ndarray) -> float:
    """
    Compare HOG features using Euclidean distance.
    """
    return euclidean(hog1, hog2)

# klopt Colorhistogram nog als afbeeldingen niet dezelfde dimensies hebben. Ja, want we normaliseren de histogrammen
def calc_histogram(img: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Calculate histograms for the blue, green, and red channels of an image.
    """
    hist_blue = cv.calcHist([img], [0], None, [256], [0, 256])
    hist_green = cv.calcHist([img], [1], None, [256], [0, 256])
    hist_red = cv.calcHist([img], [2], None, [256], [0, 256])

    return hist_blue, hist_green, hist_red


def plot_histograms(histogram1, histogram2) -> None:
    """
    Plot histograms for comparison.
    """
    # give plot name histogram 1
    plt.title("Histogram 1")
    plt.plot(histogram1[0], color='b')
    plt.plot(histogram1[1], color='g')
    plt.plot(histogram1[2], color='r')
    plt.show()

    plt.title("Histogram 2")
    plt.plot(histogram2[0], color='b')
    plt.plot(histogram2[1], color='g')
    plt.plot(histogram2[2], color='r')
    plt.show()


def get_final_score(scores: list[float]) -> float:
    """
    Creating a weighted average of the scores
    """
    return np.average(scores, weights=[0.8, 0.2])


# def affine_testje() -> None:
#     import cv2
#     import numpy as np

#     # Load the input image
#     img = cv2.imread(folder + "/" + files[0])

#     rows, cols = img.shape[:2]

#     # Define the vertices of the parallelogram
#     pt1 = np.float32([0, 0])
#     pt2 = np.float32([cols - 1, 0])
#     pt3 = np.float32([cols*0.25, rows - 1])
#     pt4 = np.float32([cols*0.75, rows - 1])

#     # Compute the transformation matrix
#     src_pts = np.array([pt1, pt2, pt3, pt4], dtype=np.float32)
#     dst_pts = np.array([[0, 0], [cols - 1, 0], [0, rows - 1],
#                        [cols - 1, rows - 1]], dtype=np.float32)
#     M = cv2.getPerspectiveTransform(src_pts, dst_pts)

#     # Apply the transformation to the image
#     result = cv2.warpPerspective(img, M, (cols, rows))
#     result = cv2.rotate(result, cv2.ROTATE_90_CLOCKWISE)

#     # Display the result
#     cv2.imwrite('output_image.jpg', result)


def scale_and_display(img: np.ndarray) -> None:
    """
    Scale down an image and display it.
    """
    scale_percent = 30  # percent of original size
    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)
    dim = (width, height)
    resized = cv.resize(img, dim, interpolation=cv.INTER_AREA)
    # plt.imshow(img3),plt.show()
    display("Image", resized)


def lowe_test(matches: list[cv.DMatch]) -> list[cv.DMatch]:
    # Apply ratio test
    """
    What is Lowe's ratio test? 

    Short version: each keypoint of the first image is matched with a number of keypoints from the second image. We keep the 2 best matches for each keypoint (best matches = the ones with the smallest distance measurement). Lowe's test checks that the two distances are sufficiently different. If they are not, then the keypoint is eliminated and will not be used for further calculations.
    """
    good = []
    lowe_ratio = 0.80
    for m, n in matches:
        if m.distance < lowe_ratio*n.distance:
            good.append([m])
    return good


def calculate_homograhpy(good: list[cv.DMatch], kp1: tuple[list[cv.KeyPoint], list[cv.KeyPoint]], kp2: tuple[list[cv.KeyPoint], list[cv.KeyPoint]]) -> float:
    """
    Calculates the homography matrix between two sets of keypoints.
    It's used to calculate the inlier ratio to determine how good the keypoints match.
    """
    # minimum number of matches required to estimate transformation matrix
    MIN_MATCH_COUNT = 10
    if len(good) >= MIN_MATCH_COUNT:
        src_pts = np.float32(
            [kp1[m[0].queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst_pts = np.float32(
            [kp2[m[0].trainIdx].pt for m in good]).reshape(-1, 1, 2)

        # Estimate transformation matrix using RANSAC
        M, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC, 5.0)

        # Count inliers
        inliers = np.sum(mask)

        # Compute ratio of inliers
        inlier_ratio = inliers / len(good)
    else:
        # Not enough matches were found, so the score is 0
        inlier_ratio = 0

    return inlier_ratio

def normalize(scores: list[float]) -> list[float]:
    """
    Normalize the scores to a value between 0 and 1.
    """
    min_score = min(scores)
    max_score = max(scores)
    return [(score - min_score) / (max_score - min_score) for score in scores]


def calculate_score_assignment2(img, db_folder):
    
    files = os.listdir(db_folder)
    img = cv.resize(img, (500, 500))
    histogram_scores = []
    matcher_scores = []
        
        
    for file2 in files:
        img2 = cv.imread(db_folder + "/" + file2)
        
        img2 = cv.resize(img2, (500, 500))

        # Initiate ORB detector, argument for number of keypoints
        orb = cv.ORB_create(500)

        # find the keypoints with ORB
        kp1, des1 = orb.detectAndCompute(img, None)
        kp2, des2 = orb.detectAndCompute(img2, None)

        # BFMatcher with default params
        bf = cv.BFMatcher()
        matches = bf.knnMatch(des1, des2, k=2)

        good = lowe_test(matches)

        score_matcher = calculate_homograhpy(good, kp1, kp2)
        matcher_scores.append(score_matcher)

        hist_blue1, hist_green1, hist_red1 = calc_histogram(img)
        hist_blue2, hist_green2, hist_red2 = calc_histogram(img2)

        # plot_histograms([hist_blue1, hist_green1, hist_red1], [hist_green2, hist_blue2, hist_red2])

        histogram_score = compare_histograms([hist_blue1, hist_green1, hist_red1], [hist_blue2, hist_green2, hist_red2])
        histogram_scores.append(histogram_score)
        

        # et = time.time()
        # elapsed_time = et - st
        # print('Execution time:', elapsed_time, 'seconds')
        
    final_scores = []
    for i in range(len(histogram_scores)):
        final_score = get_final_score(np.array([matcher_scores[i], 1 - histogram_scores[i]]))
        final_scores.append(final_score)
    
    return final_scores, files

















if __name__ == "__main__":
    folder = "Database_paintings/Database"
    



