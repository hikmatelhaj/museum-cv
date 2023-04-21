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
    return np.average(scores, weights=[0.5, 0.25, 0.25])


def affine_testje() -> None:
    import cv2
    import numpy as np

    # Load the input image
    img = cv2.imread(folder + "/" + files[0])

    rows, cols = img.shape[:2]

    # Define the vertices of the parallelogram
    pt1 = np.float32([0, 0])
    pt2 = np.float32([cols - 1, 0])
    pt3 = np.float32([cols*0.25, rows - 1])
    pt4 = np.float32([cols*0.75, rows - 1])

    # Compute the transformation matrix
    src_pts = np.array([pt1, pt2, pt3, pt4], dtype=np.float32)
    dst_pts = np.array([[0, 0], [cols - 1, 0], [0, rows - 1],
                       [cols - 1, rows - 1]], dtype=np.float32)
    M = cv2.getPerspectiveTransform(src_pts, dst_pts)

    # Apply the transformation to the image
    result = cv2.warpPerspective(img, M, (cols, rows))
    result = cv2.rotate(result, cv2.ROTATE_90_CLOCKWISE)

    # Display the result
    cv2.imwrite('output_image.jpg', result)


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

if __name__ == "__main__":
    folder = "Database_paintings/Database"
    files = os.listdir(folder)
    for file1 in files:
        img = cv.imread(folder + "/" + file1)
        img = cv.resize(img, (500, 500))
        hog_scores = []
        histogram_scores = []
        matcher_scores = []
        
        
        
        for file2 in files:
            # st = time.time()
            # print(folder + "/" + file1)
            img2 = cv.imread(folder + "/" + file2)
            img2 = cv.resize(img2, (500, 500))
            # img2 = cv.imread('output_image.jpg')

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
            


            # ideeen: structuur foto --> hog, grootte kader
            # hog
            fd, hog_image = hog(img, orientations=8, pixels_per_cell=(16, 16), cells_per_block=(1, 1), visualize=True, channel_axis=-1)
            fd2, hog_image2 = hog(img2, orientations=8, pixels_per_cell=(16, 16), cells_per_block=(1, 1), visualize=True, channel_axis=-1)
            # plt.axis("off")
            # plt.imshow(hog_image, cmap="gray")
            # plt.show()
            hog_scores.append(compare_hog_features(fd, fd2))

            # et = time.time()
            # elapsed_time = et - st
            # print('Execution time:', elapsed_time, 'seconds')
        
        final_scores = []
        print(hog_scores)
        normalised_hog = normalize(hog_scores)
        for i in range(len(hog_scores)):
            
            final_score = get_final_score(np.array([matcher_scores[i], 1 - histogram_scores[i], 1 - normalised_hog[i]]))
            final_scores.append(final_score)
        # Save results to files
        with open('matcher_scores.txt', 'w') as f:
            for item in matcher_scores:
                f.write("%s\n" % item)
                
        with open('normalised_hog.txt', 'w') as f:
            for item in normalised_hog:
                f.write("%s\n" % item)
                
        with open('histogram_scores.txt', 'w') as f:
            for item in histogram_scores:
                f.write("%s\n" % item)
                
        with open('final_scores.txt', 'w') as f:
            for item in final_scores:
                f.write("%s\n" % item)
        print("The average score is", np.mean(final_scores))
        print("All the other scores are", final_scores)
        break


"""
TODO:
testen op fotos die niet schoon zijn

Vragenlijst:
1. Wat is de beste metriek om matches te vergelijken? (# good matches en dan normalizen of met ransac  (tranform M en dan inliers) ?)
2. Hog, afbeelding rescalen, kunnen we dat doen?
3. Moet de paper in latex geschreven worden? ja
4. Misschien video sample rate vraag
5. Als we elke foto in de database vergelijken met alle andere fotos in de database, dan duurt dat super lang. Is dit normaal?
Execution time 1 foto: 0.8440244197845459 seconds => valt mee, niet alle fotos met alle fotos in db vergelijken, maar enkel met naburige 


6. hidden markov modellen

Methode om probabiliteiten en trnasities toe te wijzen:

Als je een minuut geen schilderij ziet, dan moet de probabilieit over de gangen uitwaaien, 
Probabiliteiten dan samentrekken nadat je een schilderij ziet ( e.g. p(e|x), event kies je zelf, kan bijvoorbeeld schilderij nummer 322 > score: 0.7  )
Probabiliteit berekenen van een bepaald schilderij met een ruimte


"""
