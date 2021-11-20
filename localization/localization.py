import cv2
import numpy as np
import geopy.distance
import matplotlib.pyplot as plt
import gmplot
from config import data_dir, api_key

def plot(locations, directions, l = 20):
    pid = np.arange(len(locations))

    origin = locations[0]
    pano_coords = [np.zeros(2)]
    dir_coords = [[pano_coords[0] - (directions[0] * l), pano_coords[0] + (directions[0] * l)]]
    for dir, loc in zip(directions[1:], locations[1:]):
        dy = geopy.distance.distance(origin, (loc[0], origin[1])).m
        dx = geopy.distance.distance(origin, (origin[0], loc[1])).m
        coord = np.array([dx, dy])
        pano_coords.append(coord)
        dir_coords.append([coord - (dir * l), coord + (dir * l)])
    pano_coords = np.array(pano_coords)
    dir_coords = np.array(dir_coords)
    plt.scatter(pano_coords[:, 0], pano_coords[:, 1])
    for xi, yi, pidi in zip(pano_coords[:, 0],pano_coords[:, 1],pid):
        plt.annotate(str(pidi), xy=(xi,yi))
    for dir in dir_coords:
        plt.plot(dir[:, 0], dir[:, 1])
    plt.show()

def find_homography(points1, points2, K_phone):
    K_streetview = K_phone
    K_streetview[:,-1] = 0 # reset principal point
    points1, points2 = np.array(points1), np.array(points2)
    # points1 = cv2.undistortPoints(points1, K_phone, None).reshape((-1, 2))
    # points2 = cv2.undistortPoints(points2, K_streetview, None).reshape((-1, 2))
    E, mask = cv2.findEssentialMat(points1, points2, cameraMatrix=K_phone, method=cv2.RANSAC)
    points, R, t, mask = cv2.recoverPose(E, points1, points2, K_phone, mask=mask)
    return R, -np.squeeze(t)

def estimate_location_two_panoramas(locations, directions, heading):
    origin = np.array([0, 0])
    
    dy = geopy.distance.distance(locations[0], (locations[1, 0], locations[0, 1])).m
    dx = geopy.distance.distance(locations[0], (locations[0, 0], locations[1, 1])).m
    coord = np.array([dx, dy])
    
    direction_vector_1 = [origin, origin + directions[0]]
    direction_vector_2 = [coord, coord + directions[1]]
    
    t, s = np.linalg.solve(np.array([direction_vector_1[1]-direction_vector_1[0], direction_vector_2[0]-direction_vector_2[1]]).T, direction_vector_2[0]-direction_vector_1[0])
    offset = (1-t)*direction_vector_1[0] + t*direction_vector_1[1]
    mag = np.linalg.norm(offset)
    bearing = np.arctan(offset[1]/offset[0]) + heading

    localized_coord = geopy.distance.distance(meters=mag).destination(locations[0], bearing=bearing)
    # gmap3 = gmplot.GoogleMapPlotter(34.060458, -118.437621, 17, apikey=api_key)
    # print(locations)
    # gmap3.scatter(locations[:,0], locations[:,1], '#FF0000', size=5, marker=True)
    # gmap3.scatter([localized_coord.latitude], [localized_coord.longitude], '#0000FF', size=7, marker=True)
    # gmap3.draw(f"{data_dir}/image_locations.html")

    return (localized_coord.latitude, localized_coord.longitude)