import cv2
import numpy as np
import geopy.distance
import matplotlib.pyplot as plt
import gmplot
from config import data_dir, api_key

from PIL import Image as PImage
import plotly.graph_objects as go
from plotly import tools

def create_surface(rgb, depth, max_depth=1000, **kwargs):
    rgb = rgb.swapaxes(0, 1)[:, ::-1]
    depth = depth.swapaxes(0, 1)[:, ::-1]
    eight_bit_img = PImage.fromarray(rgb).convert('P', palette='WEB', dither=None)
    idx_to_color = np.array(eight_bit_img.getpalette()).reshape((-1, 3))
    colorscale=[[i/255.0, "rgb({}, {}, {})".format(*rgb)] for i, rgb in enumerate(idx_to_color)]
    depth_map = depth.copy().astype('float')
    depth_map[depth_map>max_depth] = np.nan
    return go.Surface(
        z=depth_map,
        surfacecolor=np.array(eight_bit_img),
        cmin=0, 
        cmax=255,
        colorscale=colorscale,
        showscale=False,
        **kwargs
    )

def show_rgbd(rgb, depth):
    fig = go.Figure(
    data=[create_surface(rgb, 
                             depth,
                             contours_z=dict(show=True, project_z=True, highlightcolor="limegreen"),
                             opacity=1.0
                            )],
        layout_title_text="3D Surface"
    )
    fig.show()

def find_homography(points1, points2, K_phone):
    K_streetview = K_phone
    K_streetview[:,-1] = 0 # reset principal point
    points1, points2 = np.array(points1), np.array(points2)
    points1 = cv2.undistortPoints(points1, K_phone, None).reshape((-1, 2))
    points2 = cv2.undistortPoints(points2, K_streetview, None).reshape((-1, 2))
    E, mask = cv2.findEssentialMat(points1, points2, cameraMatrix=np.eye(3), method=cv2.RANSAC)
    points, R, t, mask = cv2.recoverPose(E, points1, points2, np.eye(3), mask=mask)
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
    gmap3 = gmplot.GoogleMapPlotter(34.060458, -118.437621, 17, apikey=api_key)
    gmap3.scatter(locations[:,0], locations[:,1], '#FF0000', size=5, marker=True)
    gmap3.scatter([localized_coord.latitude], [localized_coord.longitude], '#0000FF', size=7, marker=True)
    gmap3.draw(f"{data_dir}/image_locations.html")

    return (localized_coord.latitude, localized_coord.longitude)

def estimate_location_panorama_depth(location, t, heading, depth):
    pass