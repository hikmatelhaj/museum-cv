# imports
import cv2 as cv
import numpy as np
import pandas as pd
import glob
import ast
from shapely.geometry import Polygon
import os

class Extractor:
    """
    Extracts paintings out of images like a magician
    """

    def find_painting(self, image):
        """
        Extract painting polygons out of the image
        """
        gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        ret, th = cv.threshold(gray, 0, 255, cv.THRESH_BINARY+cv.THRESH_OTSU)
        blur = cv.GaussianBlur(th,(5,5),0)
        edges = cv.Canny(blur, 50, 120)
        dilated = cv.dilate(edges, np.ones((5, 5), np.uint8), iterations=3)
    
        contours = cv.findContours(dilated, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)[0]
        contours = sorted(contours, key=cv.contourArea, reverse=True)[:10]
        
        cut_list = []
        for c in contours:
            polygon = cv.approxPolyDP(c, 0.05*cv.arcLength(c, True), True)
            if len(polygon) == 4:
                polygon = polygon.reshape(4, 2)
                if Polygon(polygon).area > 15000:
                    image = cv.polylines(image, [polygon], True, (0, 0, 255), 10)
                    cut_list.append(polygon)
        return cut_list
    
    def crop_to_painting(self, image, polygon):
        """
        Transform image based on the extracted painting polygons of the image
        """
        maxW = max(polygon[:,0]) - min(polygon[:,0])
        maxH = max(polygon[:,1]) - min(polygon[:,1])
        bounding_box = np.array([
            [0, 0],
            [0, maxH],
            [maxW, maxH],
            [maxW, 0]], dtype="float32")
        transform = cv.getPerspectiveTransform(np.float32(polygon), bounding_box)
        result = cv.warpPerspective(image, transform, (maxW, maxH))
        return result
    
    # process single image and transform to cropped image
    def extract_and_crop_image(self, image, drawPolygons=True):
        """
        Process a single images, extracts all paintings 
        and creates new images, each with the cropped painting
        """
        polygons = self.find_painting(image)

        result_imgs = []

        for polygon in polygons:
            if drawPolygons:
                img = cv.polylines(img, [polygon], True, (0, 0, 255), 10)
            
            maxW = max(polygon[:,0]) - min(polygon[:,0])
            maxH = max(polygon[:,1]) - min(polygon[:,1])
            bounding_box = np.array([
                [0, 0],
                [0, maxH],
                [maxW, maxH],
                [maxW, 0]], dtype="float32")
            transform = cv.getPerspectiveTransform(np.float32(polygon), bounding_box)
            result = cv.warpPerspective(img, transform, (maxW, maxH))
            result_imgs.append(result)
        
        return result_imgs
