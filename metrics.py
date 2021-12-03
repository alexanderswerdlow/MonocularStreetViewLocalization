from numpy.random import rand
import geopy.distance
import numpy as np
from matplotlib import pyplot as plt
from numpy import arccos, array, dot, pi, cross
from numpy.linalg import det, norm

import gmplot
from config import api_key, data_dir

def distance_to_line(A, B, P):
        A = array(A)
        B = array(B)
        P = array(P)

        """ segment line AB, point P, where each one is an array([x, y]) """
        if all(A == P) or all(B == P):
            return 0
        if arccos(dot((P - A) / norm(P - A), (B - A) / norm(B - A))) > pi / 2:
            return norm(P - A)
        if arccos(dot((P - B) / norm(P - B), (A - B) / norm(A - B))) > pi / 2:
            return norm(P - B)
        return norm(cross(A-B, A-P))/norm(B-A)

def convert_to_meters(origin, loc):
    dx = geopy.distance.distance(origin, (loc[0], origin[1])).m
    dy = geopy.distance.distance(origin, (origin[0], loc[1])).m
    return [dx, dy]

def calculate_error(sparse_true_coords, estimated_coords):
    dist = np.zeros(len(estimated_coords))
    j = 0
    for z in estimated_coords:
        min_dist = np.inf
        tmp_dist = np.inf
        for i in range(0,len(sparse_true_coords)-1):
            tmp_dist = distance_to_line([0,0], convert_to_meters(sparse_true_coords[i], sparse_true_coords[i+1]), convert_to_meters(sparse_true_coords[i], z))
            if tmp_dist < min_dist:
                min_dist = tmp_dist
            dist[j] = min_dist
        j = j+1
        error = np.sum(dist)
    print(dist)
    avg_error = error/len(estimated_coords)
    return dist, avg_error

trajectory = np.array([(34.057658, -118.448004), (34.058030, -118.446574), (34.059171, -118.442295), (34.060069, -118.438973)])
estimated = np.array([(34.058067, -118.446948), (34.058067, -118.446948), (34.058112, -118.446658), (34.058216, -118.445938), (34.058553, -118.444983), (34.058657, -118.444255), (34.058524, -118.445055)])


dist, avg_error = calculate_error(trajectory, estimated)
x = list(range(0,len(estimated)))
plt.title("Error per point") 
plt.xlabel("point number") 
plt.ylabel("error") 
plt.plot(x,dist) 
plt.show()

# localized_coord = geopy.distance.distance(meters=mag).destination(locations[0], bearing=np.rad2deg(bearing))
gmap3 = gmplot.GoogleMapPlotter(34.1231, -118.1232, 17, apikey=api_key)
gmap3.scatter(trajectory[:,0], trajectory[:,1], '#0000FF', size=5, marker=True)
gmap3.scatter(estimated[:,0], estimated[:,1], '#FF0000', size=5, marker=True)
# gmap3.scatter([localized_coord.latitude], [localized_coord.longitude], '#0000FF', size=7, marker=True)
gmap3.draw(f"{data_dir}/trajectory_offsets.html")