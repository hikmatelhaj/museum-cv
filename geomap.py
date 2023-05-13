import pandas as pd
import geopandas as gpd
import cv2 as cv
import matplotlib.pyplot as plt
from shapely.geometry import Polygon
import numpy as np
import matplotlib.pylab as pl
from matplotlib.colors import ListedColormap
from csv import writer

import matplotlib.image as image
import folium
from shapely import wkt



# polygons = []
# coords = []

# def click_event(event, x, y, flags, params):
#     if event == cv.EVENT_LBUTTONDOWN:
#         coords.append((x,y))


# # read the input image
# img = cv.imread('msk_floorplan.png')

# img = cv.resize(img, (int(img.shape[1]*0.8), int(img.shape[0]*0.8)), interpolation= cv.INTER_LINEAR)

# # create a window
# cv.namedWindow('Point Coordinates')

# # bind the callback function to window
# cv.setMouseCallback('Point Coordinates', click_event, coords)

# # display the image

# cv.imshow('Point Coordinates', img)
# k= cv.waitKey(0)
 
# if k == 27:   # wait for esckey to exit
#     cv.destroyAllWindows()
 
# elif k == ord('p'):  # wait for p key to save and exit
#     listToStr = ",".join(map(str,coords))
#     listToStr = "POLYGON " + listToStr
#     polygons.append(listToStr)
#     coords = []
#     print(polygons)

      
#     with open('coords.csv', 'a') as f_object:
 
#         # Pass this file object to csv.writer()
#         # and get a writer object
#         writer_object = writer(f_object)
    
#         # Pass the list as an argument into
#         # the writerow()
#         writer_object.writerow(polygons)
    
#         # Close the file object
#         f_object.close()


stats = pd.read_csv("coords.csv", delimiter=";")
stats['geometry'] = stats['geometry'].apply(wkt.loads)

# Choose colormap
cmap = pl.cm.PuBu

# Get the colormap colors
my_cmap = cmap(np.arange(cmap.N))

# Set alpha
my_cmap[:,-1] = np.linspace(0, 1, cmap.N)

# Create new colormap
my_cmap = ListedColormap(my_cmap)

stats = gpd.GeoDataFrame(stats, crs="EPSG:4326", geometry="geometry")
print(stats)
fig, ax = plt.subplots(1, figsize=(8, 8))
plt.xticks(rotation=90)
stats.plot(column="probability", cmap=my_cmap, vmin=0, vmax=1, linewidth=10, ax=ax, aspect=1, alpha = 0.8)
plan = image.imread("msk_floorplan.png")
plan = cv.resize(plan, (int(plan.shape[1]*0.8), int(plan.shape[0]*0.8)), interpolation=cv.INTER_LINEAR)
plt.imshow(plan)
bar_info = plt.cm.ScalarMappable(cmap=my_cmap, norm=plt.Normalize(vmin=0, vmax=1))
bar_info._A = []
cbar = fig.colorbar(bar_info)
plt.show()






