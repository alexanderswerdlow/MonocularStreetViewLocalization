import cv2
import numpy as np
import geopy.distance
import matplotlib.pyplot as plt
import gmplot
from scipy.spatial.transform import Rotation as R

from config import images_dir


def find_correspondence_set_intersection(all_matches):
    intersection_frame_points = set(all_matches[0][0])
    all_frame_points = set(all_matches[0][0])
    for frame_points, pano_points in all_matches:
        intersection_frame_points.intersection_update(frame_points)
        all_frame_points = all_frame_points.union(frame_points)
    print(f'Frame points: {len(intersection_frame_points)}, total unique points: {len(all_frame_points)}')

    intersection_frame_points = list(intersection_frame_points)
    all_filtered_pano_points = []

    for frame_points, pano_points in all_matches:
        filtered_pano_points = []
        for frame_point in intersection_frame_points:
            filtered_pano_points.append(pano_points[np.where(np.all(np.array(frame_points) == frame_point, axis=1))[0][0]])
        all_filtered_pano_points.append(filtered_pano_points)

    return intersection_frame_points, all_filtered_pano_points

def estimate_pose_with_3d_points_old(frame_points, pano_points, locations, heading, pitch, height, K_phone):
    K_streetview = K_phone
    K_streetview[:,-1] = 0 # reset principal point

    A = np.zeros((len(pano_points[0]), len(locations)*2, 4))

    points = []
    P = []
    for i in range(len(locations)):
        dy = geopy.distance.distance(locations[0], (locations[i, 0], locations[0, 1])).m
        dx = geopy.distance.distance(locations[0], (locations[0, 0], locations[i, 1])).m

        projection = np.zeros((3, 4))
        rotation = R.from_euler('xyz', [pitch, -heading, 0], degrees=True).as_matrix() # init to just rotation matrix for now
        translation = np.array([dx, height, dy])
        projection[:3,:3] = rotation
        projection[:3,-1] = translation
        P.append(projection)
        # projection = np.matmul(K_streetview, projection)

        norm_points = cv2.undistortPoints(np.array(pano_points[i]).astype(np.float32), K_streetview, None).reshape((-1, 2))
        points.append(norm_points)

    points = np.array(points)
    X = cv2.triangulatePoints(P[0], P[1], points[0].T, points[1].T)
    X /= X[3]
    X1 = P[0][:3] @ X
    X2 = P[1][:3] @ X

    # print(X.T)
    # print(X1.T)
    # print(X2.T)

    ret, rvecs, tvecs = cv2.solvePnP(X1.T, np.array(frame_points).astype(np.float32), K_phone, None)
    print(tvecs)

    return X

def correspondence_error(p, K, y, x):
    # computes the bearing and azimuthal angles from camera pose p to feature yj in the camera frame
    # p is the camera pose
    # K is the camera intrinsics matrix
    # y is 3d object point
    # x is 2d feature image point

    # find z and z_hat and return the distance (dot product?)
    pass

def triangulation_error(P, K, y, pano_points):
    total_error = 0
    for i, p in enumerate(P):
        image_points = pano_points[i]
        for image_point in image_points:
            error = correspondence_error(p, K, y[i], image_point)
            total_error += error

    return total_error

def estimate_pose_with_3d_points(frame_points, pano_points, locations, heading, pitch, height, K_phone):
    K_streetview = K_phone
    K_streetview[:,-1] = 0 # reset principal point

    P = []

    for i in range(len(locations)):
        dy = geopy.distance.distance(locations[0], (locations[i, 0], locations[0, 1])).m
        dx = geopy.distance.distance(locations[0], (locations[0, 0], locations[i, 1])).m

        pose = np.zeros((3, 4))
        rotation = R.from_euler('xyz', [pitch, -heading, 0], degrees=True).as_matrix() # init to just rotation matrix for now
        translation = np.array([dx, height, dy])
        pose[:3,:3] = rotation
        pose[:3,-1] = translation
        P.append(pose)

        

        
    return None


def find_homography(points1, points2, K_phone, im1, im2):
    K_streetview = K_phone
    K_streetview[:,-1] = 0 # reset principal point
    points1, points2 = np.array(points1), np.array(points2)
    points1_ud = cv2.undistortPoints(points1, K_phone, None).reshape((-1, 2))
    points2_ud = cv2.undistortPoints(points2, K_streetview, None).reshape((-1, 2))

    E, mask = cv2.findEssentialMat(points2, points1, cameraMatrix=K_streetview, method=cv2.RANSAC)
    points, R, t, mask = cv2.recoverPose(E, points2, points1, K_streetview, mask=mask)

    return R, np.squeeze(t)


def estimate_location(locations, angles, l=10):
    directions = np.dstack((np.cos(angles), np.sin(angles)))[0]
    origin = np.array([0, 0])
    plt.scatter(origin[0], origin[0])

    prev_direction_vector = np.array([origin, origin + l * directions[0]])
    plt.plot(prev_direction_vector[:,0], prev_direction_vector[:,1])
    
    intersections = []
    for i in range(1, len(locations)):
        dy = geopy.distance.distance(locations[0], (locations[i, 0], locations[0, 1])).m
        dx = geopy.distance.distance(locations[0], (locations[0, 0], locations[i, 1])).m
        coord = np.array([dx, dy])
        
        direction_vector = np.array([coord, coord + l * directions[i]])
        
        t, s = np.linalg.solve(np.array([direction_vector[1]-direction_vector[0], prev_direction_vector[0]-prev_direction_vector[1]]).T, prev_direction_vector[0]-direction_vector[0])
        offset = (1-s)*direction_vector[0] + s*direction_vector[1]
        mag = np.linalg.norm(offset)
        bearing = np.arctan(offset[1]/offset[0])

        plt.scatter(coord[0], coord[1])
        # plt.scatter(offset[0], offset[1], marker='*')
        plt.plot(direction_vector[:,0], direction_vector[:,1])
    plt.show()

    localized_coord = geopy.distance.distance(meters=mag).destination(locations[0], bearing=bearing)
    # gmap3 = gmplot.GoogleMapPlotter(34.060458, -118.437621, 17, apikey=api_key)
    # gmap3.scatter(locations[:,0], locations[:,1], '#FF0000', size=5, marker=True)
    # gmap3.scatter([localized_coord.latitude], [localized_coord.longitude], '#0000FF', size=7, marker=True)
    # gmap3.draw(f"{data_dir}/image_locations.html")

    return (localized_coord.latitude, localized_coord.longitude)
