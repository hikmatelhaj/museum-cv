import cv2 as cv
import pickle
import numpy as np

class Rectifier:
    def __init__(self):
        self.cam_mtx = []
        self.dist_coeff = []
        self.calibrated = False

    def calibrate_by_video(self, video_path):
        """
        Calibrate the video
        """
        cap = cv.VideoCapture(video_path)
        fps = int(cap.get(cv.CAP_PROP_FPS))
        
        f = 0
        objpoints = [] 
        imgpoints = [] 

        i = 0

        while cap.isOpened():
            process = False
            ret, frame = cap.read()
            if not ret:
                break

            if f < fps:
                f += 1
            if f % fps == 0:
                gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
                lol = cv.Laplacian(gray, cv.CV_64F).var() 
                if lol > 30:
                    process = True
                    # Frame is not blurry
                else:
                    continue
            if process & (f%fps == 0):
                i+=1
                print(f"reading frame {i}")
                f = 1
                process = False

                criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
                objp = np.zeros((6*10,3), np.float32)
                objp[:,:2] = np.mgrid[0:6,0:10].T.reshape(-1,2)
                gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
                ret, corners = cv.findChessboardCorners(gray, (6,10), cv.CALIB_CB_ADAPTIVE_THRESH + cv.CALIB_CB_FAST_CHECK + cv.CALIB_CB_NORMALIZE_IMAGE + cv.CALIB_USE_LU)
                if ret == True:
                    objpoints.append(objp)
                    corners2 = cv.cornerSubPix(gray,corners, (11,11), (-1,-1), criteria)
                    imgpoints.append(corners2)

                   
        print("calibrating")
        ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None, flags=cv.CALIB_USE_LU)
        self.cam_mtx = mtx
        self.dist_coeff = dist

        self.calibrated = True
        
        cap.release()

    def save_calibration(self, filename_prefix):
        """
        Saves calibration in a file: 
        - camera matrix in a file ending with _mtx
        - distortion coëfficiënts in a file ending with _dist
        @param filename_prefix used as a prefix to save the calibration 
        """
        with open(f"./calibration/{filename_prefix}_mtx", "wb") as mtx_file:
            pickle.dump(self.cam_mtx, mtx_file)
        with open(f"./calibration/{filename_prefix}_dist", "wb") as dist_file:
            pickle.dump(self.dist_coeff, dist_file)

    def load_calibration(self, filename_prefix):
        """
        Loads the calibration from files saved by `save_calibration`
        @param filename_prefix used as prefix to load the files
        """
        with open(f"./calibration/{filename_prefix}_mtx", "rb") as mtx_file:
            self.cam_mtx = pickle.load(mtx_file)
        with open(f"./calibration/{filename_prefix}_dist", "rb") as dist_file:
            self.dist_coeff = pickle.load(dist_file)
        self.calibrated = True
        
    def process_image(self, image):
        """
        Distort a single image
        """
        if not self.calibrated:
            raise Exception("Not calibrated")
        return cv.undistort(image, self.cam_mtx, self.dist_coeff)
        