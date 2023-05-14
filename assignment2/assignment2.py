# https://docs.opencv.org/4.x/d1/d89/tutorial_py_orb.html

# https://stackoverflow.com/questions/39527947/how-to-calculate-score-from-orb-algorithm

import os
import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
from skimage.feature import hog
from scipy.spatial.distance import euclidean
import time
import pickle
import concurrent.futures


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


def create_keypoints_and_color_hist_db(db_folder):
    files = os.listdir(db_folder)
    kps = []
    dess = []
    histogram = []
    
    for file2 in files:
        img2 = cv.imread(db_folder + "/" + file2)
        img2 = cv.resize(img2, (500, 500))
        img2 = cv.GaussianBlur(img2, (5, 5), 0)
        
        img = cv.imread(db_folder + "/zaal_1__IMG_20190323_111717__01.png")
        orb = cv.ORB_create(500)
        kp1, des1 = orb.detectAndCompute(img, None)
        display("d", img2)

        # Initiate ORB detector, argument for number of keypoints
        orb = cv.ORB_create(100)

        # find the keypoints with ORB
        kp2, des2 = orb.detectAndCompute(img2, None)
        dess.append(des2)
        kp_list = [(k.pt, k.size, k.angle, k.response, k.octave, k.class_id) for k in kp2]
        kps.append(kp_list)

        hist_blue2, hist_green2, hist_red2 = calc_histogram(img2)
        histogram.append([hist_blue2, hist_green2, hist_red2])
        
    with open('keypoints.pkl', 'wb') as f:
        pickle.dump(kps, f)
        
    with open('descriptors.pkl', 'wb') as f:
        pickle.dump(dess, f)

    with open('histogram.pkl', 'wb') as f:
        pickle.dump(histogram, f)
    

def process_file(file, img, kp1, des1, histogram, i, kps, dess):
    # Initiate ORB detector, argument for number of keypoints
    # find the keypoints with ORB
    kp_list = kps[i]
    kp2 = [cv.KeyPoint(x, y, size, angle, response, octave, class_id) for (x, y), size, angle, response, octave, class_id in kp_list] 
    des2 = dess[i]
    bf = cv.BFMatcher()
    if des2 is None:
        print("hier", des2)
    des2 = des2.astype(np.float32)
    des1 = des1.astype(np.float32)
    # print(des1)
    # print(des2)
    matches = bf.knnMatch(des1, des2, k=2)
    good = lowe_test(matches)
    score_matcher = calculate_homograhpy(good, kp1, kp2)
    histogram_i = histogram[i]
    hist_blue1, hist_green1, hist_red1 = calc_histogram(img)
    hist_blue2, hist_green2, hist_red2 = histogram_i[0], histogram_i[1], histogram_i[2]
    histogram_score = compare_histograms([hist_blue1, hist_green1, hist_red1], [hist_blue2, hist_green2, hist_red2])
    final_score = get_final_score(np.array([score_matcher, 1 - histogram_score]))
    return final_score

def calculate_score_assignment2_multi(img, db_folder):
    st = time.time()
    files = os.listdir(db_folder)
    img = cv.resize(img, (500, 500))
    
    # Initiate ORB detector, argument for number of keypoints
    # find the keypoints with ORB
    orb = cv.ORB_create(500)
    kp1, des1 = orb.detectAndCompute(img, None)
    
    histogram_scores = []
    matcher_scores = []
    
    with open('keypoints.pkl', 'rb') as f:
        kps = pickle.load(f)

    with open('descriptors.pkl', 'rb') as f:
        dess = pickle.load(f)
        
    with open('histogram.pkl', 'rb') as f:
        histogram = pickle.load(f)

    final_scores = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        futures = []
        for i, file in enumerate(files):
            future = executor.submit(process_file, file, img, kp1, des1, histogram, i, kps, dess)
            futures.append(future)
        for future in concurrent.futures.as_completed(futures):
            final_scores.append(future.result())

    et = time.time()
    elapsed_time = et - st
    print('Execution time:', elapsed_time, 'seconds')
    return final_scores, files   

def calculate_score_assignment2(img, db_folder):
    st = time.time()
    files = os.listdir(db_folder)
    img = cv.resize(img, (500, 500))
    
    # Initiate ORB detector, argument for number of keypoints
    # find the keypoints with ORB
    orb = cv.ORB_create(500)
    kp1, des1 = orb.detectAndCompute(img, None)
    
    histogram_scores = []
    matcher_scores = []
    
    with open('keypoints.pkl', 'rb') as f:
        kps = pickle.load(f)

    with open('descriptors.pkl', 'rb') as f:
        dess = pickle.load(f)
        
    with open('histogram.pkl', 'rb') as f:
        histogram = pickle.load(f)

        
    for i, file2 in enumerate(files):

        
        # kp2, des2 = orb.detectAndCompute(img2, None)
        kp_list = kps[i]
        kp2 = [cv.KeyPoint(x, y, size, angle, response, octave, class_id) for (x, y), size, angle, response, octave, class_id in kp_list] 
        des2 = dess[i]
        
        # BFMatcher with default params
        bf = cv.BFMatcher()
        matches = bf.knnMatch(des1, des2, k=2)

        good = lowe_test(matches)

        score_matcher = calculate_homograhpy(good, kp1, kp2)
        matcher_scores.append(score_matcher)

        histogram_i = histogram[i]
        hist_blue1, hist_green1, hist_red1 = calc_histogram(img)
        hist_blue2, hist_green2, hist_red2 = histogram_i[0], histogram_i[1], histogram_i[2] # calc_histogram(img2)

        # plot_histograms([hist_blue1, hist_green1, hist_red1], [hist_green2, hist_blue2, hist_red2])

        histogram_score = compare_histograms([hist_blue1, hist_green1, hist_red1], [hist_blue2, hist_green2, hist_red2])
        histogram_scores.append(histogram_score)
        

        
    final_scores = []
    for i in range(len(histogram_scores)):
        final_score = get_final_score(np.array([matcher_scores[i], 1 - histogram_scores[i]]))
        final_scores.append(final_score)
    et = time.time()
    elapsed_time = et - st
    print('Execution time:', elapsed_time, 'seconds')
    return final_scores, files

















if __name__ == "__main__":
    folder = "Database_paintings/Database"
    



