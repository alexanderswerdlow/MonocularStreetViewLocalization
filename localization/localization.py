import cv2
import numpy as np
import geopy.distance
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
from scipy.optimize import least_squares, minimize
from functools import partial

from config import api_key
from gmplot import GoogleMapPlotter


# We subclass this just to change the map type
class CustomGoogleMapPlotter(GoogleMapPlotter):
    def __init__(self, center_lat, center_lng, zoom=25, apikey='',
                 map_type='satellite'):
        super().__init__(center_lat, center_lng, zoom, apikey)

        self.map_type = map_type
        assert(self.map_type in ['roadmap', 'satellite', 'hybrid', 'terrain'])

    def write_map(self,  f):
        f.write('\t\tvar centerlatlng = new google.maps.LatLng(%f, %f);\n' %
                (self.center[0], self.center[1]))
        f.write('\t\tvar myOptions = {\n')
        f.write('\t\t\tzoom: %d,\n' % (self.zoom))
        f.write('\t\t\tcenter: centerlatlng,\n')

        # This is the only line we change
        f.write('\t\t\tmapTypeId: \'{}\'\n'.format(self.map_type))


        f.write('\t\t};\n')
        f.write(
            '\t\tvar map = new google.maps.Map(document.getElementById("map_canvas"), myOptions);\n')
        f.write('\n')

gmap3 = CustomGoogleMapPlotter(34.060458, -118.437621, 17, apikey=api_key)
gmap3_g2o = CustomGoogleMapPlotter(34.060458, -118.437621, 17, apikey=api_key)
measurements = []

def find_correspondence_set_intersection_(all_matches):
    intersection_frame_points = set(all_matches[0][0])
    all_frame_points = set(all_matches[0][0])
    for frame_points, pano_points in all_matches:
        intersection_frame_points.intersection_update(frame_points)
        all_frame_points = all_frame_points.union(frame_points)
    print(f'Frame points: {len(intersection_frame_points)}, total unique points: {len(all_frame_points)}')
    
    intersection_frame_points = list(intersection_frame_points)
    all_filtered_pano_points = []

    for frame_points, pano_points in all_matches:
        filtered_pano_points = {}
        for idx, frame_point in enumerate(all_frame_points):
            if frame_point in frame_points:
                filtered_pano_points[idx] = pano_points[np.where(np.all(np.array(frame_points) == frame_point, axis=1))[0][0]]
        all_filtered_pano_points.append(filtered_pano_points)
    return list(all_frame_points), all_filtered_pano_points

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

def correspondence_error(p, K, y, x):
    # computes the bearing and azimuthal angles from camera pose p to feature yj in the camera frame
    # p is the camera pose
    # K is the camera intrinsics matrix
    # y is 3d object point
    # x is 2d feature image point

    # find z and z_hat and return the distance (dot product?)
    # print(p.shape, y.shape, K.shape)
    x_hat = np.matmul(np.matmul(p, np.array([*y, 1]).T), K)

    dx_hat,dy_hat,dz_hat = np.linalg.solve(K, [*x_hat[:2],1])
    dx,dy,dz = np.linalg.solve(K, [*x,1])

    theta_hat = np.arccos(dz_hat / (np.sqrt(dx_hat**2 + dy_hat**2 + dz_hat**2)))
    theta = np.arccos(dz / (np.sqrt(dx**2 + dy**2 + dz**2)))

    phi_hat = np.arctan(dy_hat / dz_hat)
    phi = np.arctan(dy / dz)

    z_hat = np.array([theta_hat, phi_hat])
    z = np.array([theta, phi])

    return np.linalg.norm(z - z_hat)
    # return err.T @ np.diag([1, 1]) @ err

def tukey_biweight(x):
    t = 454669050
    if x <= t:
        return (t**2 / 6) * (1 - (1 - (x/t)**2)**3)
    else:
        return t**2 / 6

def triangulation_error(y, P, K, pano_points):
    # from localization.biweight import biweight
    total_error = 0
    for i, p in enumerate(P):
        image_points = pano_points[i]
        for j, image_point in enumerate(image_points.values()):
            total_error += correspondence_error(p, K, y[j*3:j*3+3], image_point)**2 # tukey_biweight()

    return total_error

def estimate_pose_with_3d_points(frame_points, pano_points, locations, heading, pitch, height, K_phone):
    K_streetview = K_phone
    K_streetview[:,-1] = 0 # reset principal point
    K_streetview[-1,-1] = 1

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

    # print(len(frame_points))
    objective = partial(triangulation_error, P=P, K=K_streetview, pano_points=pano_points)
    estimate = least_squares(objective, np.zeros(len(frame_points) * 3))
    object_points = np.array(estimate.x).reshape((-1, 3))
    # print(object_points.shape)

    ret, rvecs, tvecs = cv2.solvePnP(object_points, np.array(frame_points).astype(np.float32), K_phone, None)
    
    offset = np.array(tvecs).reshape(-1)[[0,1]]
    mag = np.linalg.norm(offset)
    bearing = np.arctan(offset[0]/offset[1])

    localized_coord = geopy.distance.distance(meters=mag).destination(locations[0], bearing=np.rad2deg(bearing))
    gmap3.scatter(locations[:,0], locations[:,1], '#FF0000', size=5, marker=True)
    gmap3.scatter([localized_coord.latitude], [localized_coord.longitude], '#0000FF', size=7, marker=True)
    gmap3.draw(f"tmp/image_locations.html")

    return localized_coord.latitude, localized_coord.longitude

global num_writes
num_writes = 0
def estimate_pose_with_3d_points_g2o(frame_points, pano_points, locations, heading, pitch, height, K_phone, metadata, pts3D):
    K_streetview = K_phone
    K_streetview[:,-1] = 0 # reset principal point
    # K_streetview[-1,-1] = 1

    import g2o
    optimizer = g2o.SparseOptimizer()
    solver = g2o.BlockSolverSE3(g2o.LinearSolverCholmodSE3())
    solver = g2o.OptimizationAlgorithmLevenberg(solver)
    optimizer.set_algorithm(solver)

    v_se3 = g2o.VertexSE3Expmap()
    v_se3.set_id(0)

    dy = geopy.distance.distance(locations[0], (metadata['latitude'], locations[0, 1])).m
    dx = geopy.distance.distance(locations[0], (locations[0, 0], metadata['longitude'])).m
    rotation = R.from_euler('xyz', [pitch, -heading, 0], degrees=True).as_matrix() # init to just rotation matrix for now
    translation = np.array([dx, height, dy])
    v_se3.set_estimate(g2o.SE3Quat(rotation, [0, 0, 0]))
    optimizer.add_vertex(v_se3)

    for i in range(len(locations)):
        dy = geopy.distance.distance(locations[0], (locations[i, 0], locations[0, 1])).m
        dx = geopy.distance.distance(locations[0], (locations[0, 0], locations[i, 1])).m
        pose = np.zeros((3, 4))
        rotation = R.from_euler('xyz', [pitch, -heading, 0], degrees=True).as_matrix() # init to just rotation matrix for now
        translation = np.array([dx, height, dy])
        pose[:3,:3] = rotation
        pose[:3,-1] = translation
        pose = g2o.SE3Quat(rotation, translation)

        v_se3 = g2o.VertexSE3Expmap()
        v_se3.set_id(i + 1)
        v_se3.set_estimate(pose)
        optimizer.add_vertex(v_se3)


    cam = g2o.CameraParameters(K_phone[0, 0], np.array([K_phone[0, 2], K_phone[1, 2]]), 0)
    cam.set_id(0)
    optimizer.add_parameter(cam)

    cam = g2o.CameraParameters(K_streetview[0, 0], np.array([K_streetview[0, 2], K_streetview[1, 2]]), 0)
    cam.set_id(1)
    optimizer.add_parameter(cam)

    point_id = len(locations) + 1
    for i in range(len(frame_points)):
        vp = g2o.VertexSBAPointXYZ()
        vp.set_id(point_id)
        vp.set_marginalized(True)
        vp.set_estimate(pts3D[i])
        optimizer.add_vertex(vp)

        for j in range(len(locations)):
            if i in pano_points[j]:
                edge = g2o.EdgeProjectXYZ2UV()
                edge.set_vertex(0, vp)
                edge.set_vertex(1, optimizer.vertex(j))
                edge.set_measurement(pano_points[j][i])
                edge.set_information(np.identity(2))
                rk = g2o.RobustKernelHuber()
                rk.set_delta(np.sqrt(5.99))
                edge.set_robust_kernel(rk)
                edge.set_parameter_id(0, 1)
                optimizer.add_edge(edge)


        edge = g2o.EdgeProjectXYZ2UV()
        edge.set_vertex(0, vp)
        edge.set_vertex(1, optimizer.vertex(0))
        edge.set_measurement(frame_points[i])
        edge.set_information(np.identity(2))
        rk = g2o.RobustKernelHuber()
        rk.set_delta(np.sqrt(5.99))
        edge.set_robust_kernel(rk)
        edge.set_parameter_id(0, 0)
        optimizer.add_edge(edge)
        point_id += 1

    print('num vertices:', len(optimizer.vertices()), 'num edges:', len(optimizer.edges()))

    optimizer.set_verbose(False)
    optimizer.initialize_optimization()
    optimizer.optimize(10000)
    
    object_points = np.zeros((len(frame_points), 3))
    for i in range(len(frame_points)):
        object_points[i, :] = optimizer.vertex(1 + i + len(locations)).estimate()
    
    if len(object_points) < 6:
        return None

    np.set_printoptions(suppress=True)
    print(pts3D)
    print(np.linalg.norm(pts3D - object_points))
    try:
        ret, rvecs, tvecs = cv2.solvePnP(object_points, np.array(frame_points).astype(np.float32), K_streetview, None)
    except:
        return None
    
    offset = np.array(tvecs).reshape(-1)[[0,1]]
    mag = np.linalg.norm(offset)
    bearing = np.arctan(offset[0]/offset[1])

    localized_coord = geopy.distance.distance(meters=mag).destination(locations[0], bearing=np.rad2deg(bearing))
    gmap3_g2o.scatter(locations[:,0], locations[:,1], '#FF0000', size=5, marker=True)
    gmap3_g2o.scatter([localized_coord.latitude], [localized_coord.longitude], '#0000FF', size=7, marker=True)
    global num_writes
    num_writes += 1
    if num_writes % 50 == 0:
        gmap3_g2o.draw(f"tmp/image_locations_g2o.html")

    return localized_coord.latitude, localized_coord.longitude


def find_homography(points1, points2, K_phone, im1, im2):
    K_phone[-1,-1] = 1
    K_streetview = K_phone.copy()
    # K_streetview[:,-1] = 0 # reset principal point
    points1, points2 = np.array(points1), np.array(points2)

    E, mask = cv2.findEssentialMat(points1, points2, cameraMatrix=K_streetview, method=cv2.RANSAC)
    points, R, t, mask = cv2.recoverPose(E, points1, points2, K_streetview, mask=mask)

    Pr_1 = np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0]])
    Pr_2 = np.hstack((np.dot(K_streetview,R),np.dot(K_streetview,t)))

    pts4D = cv2.triangulatePoints(Pr_1, Pr_2, points1.T, points2.T).T
    pts3D = pts4D[:, :3]/np.repeat(pts4D[:, 3], 3).reshape(-1, 3)
    return pts3D


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
