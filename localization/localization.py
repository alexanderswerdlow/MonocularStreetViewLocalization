from collections import defaultdict
from re import T
import cv2
import numpy as np
import geopy.distance
import matplotlib.pyplot as plt
import gmplot
from scipy.spatial.transform import Rotation as R
from scipy.optimize import least_squares, minimize
from functools import partial

from config import data_dir, api_key

from config import images_dir
from gmplot import GoogleMapPlotter


class CustomGoogleMapPlotter(GoogleMapPlotter):
    def __init__(self, center_lat, center_lng, zoom=17, apikey='AIzaSyBjB2MBFlKlTIAbnG8D_t1oPqfObdR0xAA',
                 map_type='satellite'):
        super().__init__(34.060458, -118.437621, 17, apikey=api_key)

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



def find_correspondence_set_intersection(all_matches):
    intersection_frame_points = set(all_matches[0][0])
    all_frame_points = set(all_matches[0][0])
    for frame_points, pano_points in all_matches:
        intersection_frame_points.intersection_update(frame_points)
        all_frame_points = all_frame_points.union(frame_points)
    # print(f'Frame points: {len(intersection_frame_points)}, total unique points: {len(all_frame_points)}')

    intersection_frame_points = list(intersection_frame_points)
    all_filtered_pano_points = []

    for frame_points, pano_points in all_matches:
        filtered_pano_points = []
        for frame_point in intersection_frame_points:
            filtered_pano_points.append(pano_points[np.where(np.all(np.array(frame_points) == frame_point, axis=1))[0][0]])
        all_filtered_pano_points.append(filtered_pano_points)

    return intersection_frame_points, all_filtered_pano_points

def tukey_loss(x, t=6):
    if np.abs(x) <= t:
        return ((t**2)/6) * (1 - (1 - (x / t)**2)**3)
    else:
        return (t**2)/6

def correspondence_error(p, K, y, x):
    # computes the bearing and azimuthal angles from camera pose p to feature yj in the camera frame
    # p is the camera pose
    # K is the camera intrinsics matrix
    # y is 3d object point
    # x is 2d feature image point

    # find z and z_hat and return the distance (dot product?)

    x_hat = np.matmul(np.matmul(p, np.array([*y, 1]).T), K)
    return np.linalg.norm(x - x_hat[:2])

    dx_hat, dy_hat, dz_hat = np.linalg.solve(K, [*x_hat[:2], 1])
    dx, dy, dz = np.linalg.solve(K, [*x, 1])

    theta_hat = np.arccos(dz_hat / (np.sqrt(dx_hat**2 + dy_hat**2 + dz_hat**2)))
    theta = np.arccos(dz / (np.sqrt(dx**2 + dy**2 + dz**2)))

    phi_hat = np.arctan(dy_hat / dz_hat)
    phi = np.arctan(dy / dz)

    z_hat = np.array([theta_hat, phi_hat])
    z = np.array([theta, phi])

    return np.linalg.norm(z - z_hat)


def triangulation_error(y, P, K, pano_points):
    total_error = 0
    for i, p in enumerate(P):
        image_points = pano_points[i]
        for j, image_point in enumerate(image_points):
            error = correspondence_error(p, K, y[j*3:j*3+3], image_point)
            total_error += tukey_loss(error**2)

    return total_error


def estimate_pose_with_3d_points(frame_points, pano_points, locations, heading, pitch, height, K_phone):
    # 1. Find 3d coordinates from just the panoramas. Initial guess is just triangulation from 2 panos
    #    We know 6DOF pose for each pano and image points (all image points for each pano is sorted relative to its
    #    corresponding frame point), so we can calculate the 3d points (apply a solver)
    # 2. PnP solver to find frame points pose w.r.t 3d points
    K_streetview = K_phone  # .copy()
    K_streetview[:, -1] = 0  # reset principal point
    K_streetview[-1, -1] = 1

    P = []

    for i in range(len(locations)):
        dx = geopy.distance.distance(locations[0], (locations[i, 0], locations[0, 1])).m
        dy = geopy.distance.distance(locations[0], (locations[0, 0], locations[i, 1])).m

        pose = np.zeros((3, 4))
        rotation = R.from_euler('xyz', [pitch, -heading, 0], degrees=True).as_matrix()  # init to just rotation matrix for now
        translation = np.array([dx, height, dy])
        pose[:3, :3] = rotation
        pose[:3, -1] = translation
        P.append(pose)

    objective = partial(triangulation_error, P=P, K=K_streetview, pano_points=pano_points)
    try:
        estimate = least_squares(objective, np.zeros(len(frame_points) * 3))

        object_points = np.array(estimate.x).reshape((-1, 3))

        ret, rvecs, tvecs = cv2.solvePnP(object_points, np.array(frame_points).astype(np.float32), K_phone, None)
    except:
        return None, None, None, None

    reprojected_points, _ = cv2.projectPoints(object_points, rvecs, tvecs, K_phone, None)
    error = cv2.norm(np.array(frame_points), reprojected_points.reshape(-1, 2), cv2.NORM_L2)/len(reprojected_points)

    # print(np.array(tvecs).reshape(-1))
    offset = np.array(tvecs).reshape(-1)[[0, 2]]
    mag = np.linalg.norm(offset)
    bearing = np.arctan2(offset[0], offset[1])

    localized_coord = geopy.distance.distance(meters=mag).destination(locations[0], bearing=np.rad2deg(bearing))
    return estimate, error, localized_coord, locations


def find_points(points1, points2, K_phone, K_streetview):
    points1, points2 = cv2.undistortPoints(points1, K_phone, None).squeeze(), cv2.undistortPoints(points2, K_streetview, None).squeeze()
    K = np.eye(3)
    E, mask = cv2.findEssentialMat(points1, points2, cameraMatrix=K, method=cv2.RANSAC)
    points, R, t, mask = cv2.recoverPose(E, points1, points2, cameraMatrix=K, mask=mask)

    Pr_1 = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]])
    Pr_2 = np.hstack((np.dot(K, R), np.dot(K, t)))
    pts4D = cv2.triangulatePoints(Pr_1, Pr_2, points1.T, points2.T).T
    pts3D = pts4D[:, :3]/np.repeat(pts4D[:, 3], 3).reshape(-1, 3)
    return pts3D

def find_n_intersection(matches, n_required = 3):
    all_frame_points = set(matches[0][0])
    freq = defaultdict(int)
    for frame_points, pano_points in matches:
        all_frame_points = all_frame_points.union(frame_points)
        for f in frame_points:
            freq[f] += 1

    for k, v in freq.items():
        if v < 3:
            all_frame_points.remove(k)

    all_filtered_pano_points = []

    for frame_points, pano_points in matches:
        filtered_pano_points = {}
        for idx, frame_point in enumerate(all_frame_points):
            if frame_point in frame_points:
                filtered_pano_points[idx] = pano_points[np.where(np.all(np.array(frame_points) == frame_point, axis=1))[0][0]]
        all_filtered_pano_points.append(filtered_pano_points)

    return list(all_frame_points), all_filtered_pano_points

def estimate_pose_with_3d_points_g2o(matches, locations, heading, pitch, height, K_phone, metadata, fov, scaled_frame_width, scaled_frame_height):
    K_streetview = K_phone.copy()
    K_streetview[:, -1] = 0  # reset principal point
    K_streetview[-1, -1] = 1

    frame_points, pano_points = find_n_intersection(matches, n_required = 3)

    import g2o
    optimizer = g2o.SparseOptimizer()
    solver = g2o.BlockSolverSE3(g2o.LinearSolverCholmodSE3())
    solver = g2o.OptimizationAlgorithmLevenberg(solver)
    optimizer.set_algorithm(solver)

    v_se3 = g2o.VertexSE3Expmap()
    v_se3.set_id(0)

    dy = geopy.distance.distance(locations[0], (metadata['latitude'], locations[0, 1])).m
    dx = geopy.distance.distance(locations[0], (locations[0, 0], metadata['longitude'])).m
    rotation = R.from_euler('xyz', [pitch, -heading, 0], degrees=True).as_matrix()  # init to just rotation matrix for now
    translation = np.array([dx, height, dy])
    v_se3.set_estimate(g2o.SE3Quat(rotation, translation))
    optimizer.add_vertex(v_se3)

    for i in range(len(locations)):
        dy = geopy.distance.distance(locations[0], (locations[i, 0], locations[0, 1])).m
        dx = geopy.distance.distance(locations[0], (locations[0, 0], locations[i, 1])).m
        pose = np.zeros((3, 4))
        rotation = R.from_euler('xyz', [pitch, -heading, 0], degrees=True).as_matrix()  # init to just rotation matrix for now
        translation = np.array([dx, height, dy])
        pose[:3, :3] = rotation
        pose[:3, -1] = translation
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
        pt_avg = []
        for j in range(len(locations)):
            if i in pano_points[j]:
                try:
                    pts3D = find_points(np.array(matches[j][0]), np.array(matches[j][1]), K_phone, K_streetview)
                    idx = np.where(np.array(pano_points[j][i]) == np.array(matches[j][1]))[0][0]
                    if np.linalg.norm(pts3D[idx]) < 200:
                        pt_avg.append(pts3D[idx])
                except Exception as e:
                    continue

        vp = g2o.VertexSBAPointXYZ()
        vp.set_id(point_id)
        vp.set_marginalized(True)
        vp.set_estimate(np.mean(pt_avg, axis=0))
        optimizer.add_vertex(vp)

        for j in range(len(locations)):
            if i in pano_points[j]:
                edge = g2o.EdgeProjectXYZ2UV()
                edge.set_vertex(0, vp)
                edge.set_vertex(1, optimizer.vertex(j))
                edge.set_measurement(pano_points[j][i])
                edge.set_information(np.identity(2))
                rk = g2o.RobustKernelTukey()
                rk.set_delta(4.685)
                edge.set_robust_kernel(rk)
                edge.set_parameter_id(0, 1)
                optimizer.add_edge(edge)

        edge = g2o.EdgeProjectXYZ2UV()
        edge.set_vertex(0, vp)
        edge.set_vertex(1, optimizer.vertex(0))
        edge.set_measurement(frame_points[i])
        edge.set_information(np.identity(2))
        rk = g2o.RobustKernelTukey()
        rk.set_delta(4.685)
        edge.set_robust_kernel(rk)
        edge.set_parameter_id(0, 0)
        optimizer.add_edge(edge)
        point_id += 1

    # print('num vertices:', len(optimizer.vertices()), 'num edges:', len(optimizer.edges()))

    optimizer.set_verbose(False)
    optimizer.initialize_optimization()
    optimizer.optimize(10000)

    object_points = np.zeros((len(frame_points), 3))
    for i in range(len(frame_points)):
        object_points[i, :] = optimizer.vertex(1 + i + len(locations)).estimate()

    if len(object_points) < 6:
        return None, None, None, None

    np.set_printoptions(suppress=True)

    try:
        ret, rvecs, tvecs = cv2.solvePnP(object_points, np.array(frame_points).astype(np.float32), K_phone, None)
    except:
        return None, None, None, None

    offset = np.array(tvecs).reshape(-1)[[0, 1]]
    mag = np.linalg.norm(offset)
    bearing = np.arctan(offset[0]/offset[1])

    localized_coord = geopy.distance.distance(meters=mag).destination(locations[0], bearing=np.rad2deg(bearing))

    return None, None, localized_coord, locations


def estimate_pose_with_3d_points_ceres(matches, locations, heading, pitch, height, K_phone, metadata, fov, scaled_frame_width, scaled_frame_height):
    pyceres_location = "/home/aswerdlow/github/ceres-bin/lib"
    import sys
    sys.path.insert(0, pyceres_location)
    import PyCeres

    K_streetview = K_phone.copy()
    K_streetview[:, -1] = 0  # reset principal point
    K_streetview[-1, -1] = 1

    frame_points, pano_points = find_n_intersection(matches, n_required = 3)

    problem = PyCeres.Problem()

    cameras = np.zeros((len(locations)+1, 9))
    for i in range(len(locations)):
        dy = geopy.distance.distance(locations[0], (locations[i, 0], locations[0, 1])).m
        dx = geopy.distance.distance(locations[0], (locations[0, 0], locations[i, 1])).m
        pose = np.zeros(9)
        rotation = R.from_euler('xyz', [pitch, -heading, 0], degrees=True).as_mrp()  # init to just rotation matrix for now
        translation = np.array([dx, height, dy])
        cameras[i, :3] = rotation
        cameras[i, 3:6] = translation
        cameras[i, 6] = K_streetview[0, 0]

    dy = geopy.distance.distance(locations[0], (metadata['latitude'], locations[0, 1])).m
    dx = geopy.distance.distance(locations[0], (locations[0, 0], metadata['longitude'])).m
    rotation = R.from_euler('xyz', [pitch, -heading, 0], degrees=True).as_mrp()  # init to just rotation matrix for now
    translation = np.array([dx, height, dy])
    cameras[-1, :3] = rotation
    cameras[-1, 3:6] = translation
    cameras[-1, 6] = K_streetview[0, 0]

    all_pts = np.zeros((len(frame_points), 3))
    for i in range(len(frame_points)):
        for j in range(len(locations)):
            if i in pano_points[j]:
                try:
                    pts3D = find_points(np.array(matches[j][0]), np.array(matches[j][1]), K_phone, K_streetview)
                    idx = np.where(np.array(pano_points[j][i]) == np.array(matches[j][1]))[0][0]
                    all_pts[i] = pts3D[idx]
                except Exception as e:
                    print(e)
                    return None, None, None, None

                cost_function = PyCeres.CreateSnavelyCostFunction(pano_points[j][i][0], pano_points[j][i][1])
                loss = PyCeres.TukeyLoss(4.685)
                problem.AddResidualBlock(cost_function, loss, cameras[j], all_pts[i])
                for k in range(3, 6):
                    problem.SetParameterLowerBound(cameras[j], k, cameras[j][k] - 0.2)
                    problem.SetParameterUpperBound(cameras[j], k, cameras[j][k] + 0.2)

        cost_function = PyCeres.CreateSnavelyCostFunction(frame_points[i][0], frame_points[i][1])
        loss = PyCeres.TukeyLoss(4.685)
        problem.AddResidualBlock(cost_function, loss, cameras[-1], all_pts[i])

    np.set_printoptions(suppress=True)
    options = PyCeres.SolverOptions()
    options.linear_solver_type = PyCeres.LinearSolverType.DENSE_SCHUR
    options.minimizer_progress_to_stdout = False
    options.max_num_iterations = 200

    summary = PyCeres.Summary()
    PyCeres.Solve(options, problem, summary)

    tvecs = cameras[-1][3:6]
    offset = np.array(tvecs).reshape(-1)[[0, 1]]
    mag = np.linalg.norm(offset)
    bearing = np.arctan(offset[0]/offset[1])

    localized_coord = geopy.distance.distance(meters=mag).destination(locations[0], bearing=np.rad2deg(bearing))


    return None, None, localized_coord, locations