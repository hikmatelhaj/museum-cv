import cv2 as cv
import numpy as np
from shapely.geometry import Polygon


class Extractor:
    """
    Extracts paintings out of images like a magician
    """

    def find_painting(self, image: np.ndarray) -> list:
        """
        Extract painting polygons out of the image
        :return: list of polygons (extracted paintings)
        """
        gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        ret, th = cv.threshold(gray, 0, 255, cv.THRESH_BINARY+cv.THRESH_OTSU)
        blur = cv.GaussianBlur(th, (5, 5), 0)
        edges = cv.Canny(blur, 50, 120)
        dilated = cv.dilate(edges, np.ones((5, 5), np.uint8), iterations=3)

        contours = cv.findContours(
            dilated, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)[0]
        contours = sorted(contours, key=cv.contourArea, reverse=True)[:10]

        cut_list = []
        for c in contours:
            polygon = cv.approxPolyDP(c, 0.05*cv.arcLength(c, True), True)
            if len(polygon) == 4:
                # only rectangular shapes
                polygon = polygon.reshape(4, 2)
                polygonGeometry = Polygon(polygon)

                # this will check if the convex hull of the extracted polygon corresponds with the extracted polygon (so no weird shapes will be extracted)
                polygonConvexHull = polygonGeometry.convex_hull
                invalidArea = ((polygonGeometry.area + 150 < polygonConvexHull.area) or (polygonGeometry.area - 150 > polygonConvexHull.area))

                
                # too small polygons or weird shapes are filtered
                if (polygonGeometry.area > 15000) and (not invalidArea):
                    # image = cv.polylines(
                    #     image, [polygon], True, (0, 0, 255), 10)
                    cut_list.append(polygon)
        return cut_list

    def crop_to_painting(self, image: np.ndarray, polygon: np.ndarray) -> np.ndarray:
        """
        Transform image based on the extracted painting polygons of the image
        """
        maxW = max(polygon[:, 0]) - min(polygon[:, 0])
        maxH = max(polygon[:, 1]) - min(polygon[:, 1])
        bounding_box = np.array([
            [0, 0],         # top left
            [0, maxH],      # bottom left
            [maxW, maxH],   # bottom right
            [maxW, 0]],     # bottom right
            dtype="float32")

        # transform image
        transform = cv.getPerspectiveTransform(
            np.float32(polygon), bounding_box)
        result = cv.warpPerspective(image, transform, (maxW, maxH))
        return result

    def extract_and_crop_image(self, image: np.ndarray, drawPolygons: bool = True):
        """
        Process a single images, extracts all paintings 
        and creates new images, each with the cropped painting
        :return: polygons, cropped images of paintings, image with polygons drawed
        """
        # extract paintings
        polygons = self.find_painting(image)

        result_imgs = []
        img_with_polygons = image.copy()

        for polygon in polygons:
            # draw if necessary
            if drawPolygons:
                img_with_polygons = cv.polylines(img_with_polygons, [polygon], True, (0, 0, 255), 10)

            # calculate new coordinatees
            maxW = max(polygon[:, 0]) - min(polygon[:, 0])
            maxH = max(polygon[:, 1]) - min(polygon[:, 1])
            bounding_box = np.array([
                [0, 0],
                [0, maxH],
                [maxW, maxH],
                [maxW, 0]], dtype="float32")
            
            # transformation of image
            transform = cv.getPerspectiveTransform(
                np.float32(polygon), bounding_box)
            result = cv.warpPerspective(image, transform, (maxW, maxH))

            # add new painting image to result list
            result_imgs.append(result)

        return polygons, result_imgs, img_with_polygons
