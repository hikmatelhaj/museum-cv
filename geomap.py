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

def showHeatmap(probs):
    # Geometries file
    geometries = pd.read_csv("coords.csv", delimiter=";")
    stats = pd.merge(geometries,probs, on='Hall')
    stats['geometry'] = stats['geometry'].apply(wkt.loads)

    # Choose colormap
    cmap = pl.cm.PuBu

    # Get the colormap colors
    my_cmap = cmap(np.arange(cmap.N))

    # Set alpha
    my_cmap[:,-1] = np.linspace(0, 1, cmap.N)

    # Create new colormap
    my_cmap = ListedColormap(my_cmap)

    # Create heatmap
    stats = gpd.GeoDataFrame(stats, crs="EPSG:4326", geometry="geometry")
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

probabs = pd.read_csv("probs.csv", delimiter=";")
showHeatmap(probabs)



