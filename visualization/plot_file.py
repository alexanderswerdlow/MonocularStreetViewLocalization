import itertools
from numpy.random import rand
import geopy.distance
import numpy as np
from matplotlib import pyplot as plt
from numpy import arccos, array, dot, pi, cross
from numpy.linalg import det, norm
from scipy.sparse import data
import scipy.stats as stat
from scipy.stats import iqr
import gmplot
from config import api_key, end_frame, start_frame, data_dir

import pickle
compute = pickle.load(open(f"{data_dir}/gps_plot_data.pkl", "rb"))

lat0, long0 = tuple(zip(*compute[0]))
lat1, long1 = tuple(zip(*compute[1]))
lat2, long2 = tuple(zip(*compute[2]))

from localization.localization import CustomGoogleMapPlotter
gmap3 = CustomGoogleMapPlotter(34.060458, -118.437621, 17, apikey=api_key)#
# gmap3 = gmplot.GoogleMapPlotter(34.060458, -118.437621, 17, apikey=api_key)
gmap3.plot(lat0, long0, '#FF0000', ew=2, marker=True)
gmap3.plot(lat1, long1, '#0000FF', ew=2, marker=True)
gmap3.plot(lat2, long2, '#00FF00', ew=2, marker=True)
# gmap3.scatter(estimated[:,0], estimated[:,1], '#0000FF', size=5, marker=True)
gmap3.draw(f"{data_dir}/anuj.html")