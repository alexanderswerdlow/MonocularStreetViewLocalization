import cv2
import numpy as np
import geopy.distance
import matplotlib.pyplot as plt
import gmplot
from scipy.spatial.transform import Rotation as R

from config import images_dir

def drawlines(img1, img2, lines, pts1, pts2):
    
    r, c = img1.shape
    img1 = cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR)
    img2 = cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR)
      
    for r, pt1, pt2 in zip(lines, pts1, pts2):
          
        color = tuple(np.random.randint(0, 255,
                                        3).tolist())
          
        x0, y0 = map(int, [0, -r[2] / r[1] ])
        x1, y1 = map(int, 
                     [c, -(r[2] + r[0] * c) / r[1] ])
          
        img1 = cv2.line(img1, 
                        (x0, y0), (x1, y1), color, 1)
        img1 = cv2.circle(img1,
                          tuple(pt1), 5, color, -1)
        img2 = cv2.circle(img2, 
                          tuple(pt2), 5, color, -1)
    return img1, img2

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

def estimate_pose_with_3d_points(frame_points, pano_points, locations, heading, pitch, height, K_phone):
    K_streetview = K_phone
    K_streetview[:,-1] = 0 # reset principal point

    A = np.zeros((len(pano_points[0]), len(locations)*2, 4))

    for i in range(len(locations)):
        dy = geopy.distance.distance(locations[0], (locations[i, 0], locations[0, 1])).m
        dx = geopy.distance.distance(locations[0], (locations[0, 0], locations[i, 1])).m

        projection = R.from_euler('xyz', [pitch, heading, 0], degrees=True).as_matrix() # init to just rotation matrix for now
        translation = np.array([dx, dy, height]).reshape((3, 1))
        projection = np.append(projection, translation, axis=1)
        # projection = np.matmul(K_streetview, projection)

        points = cv2.undistortPoints(np.array(pano_points[i]).astype(np.float32), K_streetview, None).reshape((-1, 2))

        row2 = projection[2,:]
        for j, p in enumerate(points):
            x, y = p
            A[j, i] = x*row2 - projection[0,:]
            A[j, i+1] = y*row2 - projection[1,:]

    X = []
    for l in A:
        u,d,vt=np.linalg.svd(l)
        X.append(vt[-1,0:3]/vt[-1,3]) # normalize
    X = np.array(X)

    ret, rvecs, tvecs = cv2.solvePnP(X, np.array(frame_points).astype(np.float32), K_phone, None)
    print(tvecs)

    return X

def find_homography(points1, points2, K_phone, im1, im2):
    K_streetview = K_phone
    K_streetview[:,-1] = 0 # reset principal point
    points1, points2 = np.array(points1), np.array(points2)
    points1_ud = cv2.undistortPoints(points1, K_phone, None).reshape((-1, 2))
    points2_ud = cv2.undistortPoints(points2, K_streetview, None).reshape((-1, 2))

    E, mask = cv2.findEssentialMat(points2_ud, points1_ud, cameraMatrix=np.eye(3), method=cv2.RANSAC)
    points, R, t, mask = cv2.recoverPose(E, points2_ud, points1_ud, np.eye(3), mask=mask)

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

def bundle_adjustment():
    pass