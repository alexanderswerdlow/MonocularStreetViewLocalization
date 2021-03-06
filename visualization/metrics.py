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


def pairwise(iterable):
    "s -> (s0,s1), (s1,s2), (s2, s3), ..."
    a, b = itertools.tee(iterable)
    next(b, None)
    return zip(a, b)


def calculate_error(sparse_true_coords, estimated_coords):
    dist = []
    for idx, z in enumerate(estimated_coords):
        min_dist = np.inf
        for i, j in pairwise(sparse_true_coords):
            tmp_dist = distance_to_line([0, 0], convert_to_meters(i, j), convert_to_meters(i, z))
            if tmp_dist < min_dist:
                min_dist = tmp_dist
        dist.append(min_dist)
    return np.array(dist)


def sd_outlier(x, axis=None, bar=3, side='both'):
    assert side in ['gt', 'lt', 'both'], 'Side should be `gt`, `lt` or `both`.'

    d_z = stat.zscore(x, axis=axis)

    if side == 'gt':
        return d_z > bar
    elif side == 'lt':
        return d_z < -bar
    elif side == 'both':
        return np.abs(d_z) > bar


def sortoutOutliers(dataIn, factor):
    quant3, quant1 = np.percentile(dataIn, [75, 25])
    iqr = quant3 - quant1
    iqrSigma = iqr/1.34896
    medData = np.median(dataIn)
    dataOut = [((x > medData - factor * iqrSigma) and (x < medData + factor * iqrSigma)) for x in dataIn]
    return(dataOut)


def kalman_filter(estimated):
    from pykalman import KalmanFilter
    kf = KalmanFilter(initial_state_mean=estimated[0], n_dim_obs=2)
    return kf.em(estimated, n_iter=10).smooth(estimated)[0]

def outlier_rejection(estimated):
    a = sortoutOutliers(estimated[:, 0], 2)
    b = sortoutOutliers(estimated[:, 1], 2)
    est = []
    for i in range(len(estimated)):
        if a[i] and b[i]:
            est.append((estimated[i][0], estimated[i][1]))
    return np.array(est)

def plot_stuff(min_errors, indicies, solver):
    plt.title(f"Distance to Ground Truth, RMSE: {round(np.sqrt(np.mean(min_errors**2)), 3)}, Std: {round(min_errors.std(), 3)}")
    plt.ylim(0, 10)
    plt.xlabel("Estimated Point")
    plt.ylabel("Min Dist to Reference Trajectory (Meters)")
    plt.plot(sorted(indicies), [x for _, x in sorted(zip(indicies, min_errors))])
    plt.savefig(f'{data_dir}/error-{solver}.png', bbox_inches='tight', dpi=600)
    plt.clf()

def save_map(estimated, trajectory, solver):
    from localization.localization import CustomGoogleMapPlotter
    gmap3 = CustomGoogleMapPlotter(34.060458, -118.437621, 17, apikey=api_key)
    gmap3.plot(trajectory[:,0], trajectory[:,1], '#FF0000', size=5, marker=True)
    gmap3.scatter(estimated[:,0], estimated[:,1], '#0000FF', size=5, marker=True)
    gmap3.draw(f"{data_dir}/{solver}.html")

def process_data(data_points, solver):
    from download.waypoints import reference
    trajectory = np.array(reference)
    if len(data_points) == 0:
        return 0, None
    indices, estimated = list(zip(*[(k, (x[1].latitude, x[1].longitude)) for k,x in data_points.items()]))
    estimated = np.array(estimated)
    # if solver == 'scipy' and len(estimated) > 0:
    #     estimated = outlier_rejection(estimated)

    # try:
    import time; start_time = time.time()
    min_errors = calculate_error(trajectory, estimated)
    print(f'Metrics for {solver} took before kalman: {time.time() - start_time}')
    # kalman_estimated = kalman_filter(estimated)
    # min_errors_kalman = calculate_error(trajectory, kalman_estimated)
    print(f'Metrics for {solver} took after kalman: {time.time() - start_time}')
    plot_stuff(min_errors, indices, solver)
    # plot_stuff(min_errors_kalman, indices, solver + '-kalman')
    save_map(estimated, trajectory, solver)
    # except:
    #     breakpoint()
    return np.sqrt(np.mean(min_errors**2)), estimated

def process_data_2(estimated, indices, solver):
    from download.waypoints import reference
    trajectory = np.array(reference)
    if len(indices) == 0:
        return 0, None
    estimated = np.array(estimated)
    # breakpoint()
    # if solver == 'scipy' and len(estimated) > 0:
    #     estimated = outlier_rejection(estimated)

    # try:
    import time; start_time = time.time()
    min_errors = calculate_error(trajectory, estimated)
    print(f'Metrics for {solver} took before kalman: {time.time() - start_time}')
    # kalman_estimated = kalman_filter(estimated)
    # min_errors_kalman = calculate_error(trajectory, kalman_estimated)
    print(f'Metrics for {solver} took after kalman: {time.time() - start_time}')
    plot_stuff(min_errors, indices, solver)
    # plot_stuff(min_errors_kalman, indices, solver + '-kalman')
    save_map(estimated, trajectory, solver)
    # except:
    #     breakpoint()
    return np.sqrt(np.mean(min_errors**2)), estimated


if __name__ == '__main__':
    import pickle
    for solver in ['scipy']:
        try:
            compute = pickle.load(open(f"{data_dir}/{solver}.p", "rb"))  # compute_good
        except (OSError, IOError) as e:
            print(f"Failed to read: {solver}")
            continue
        data_points = {k: v for k, v in compute.items() if start_frame <= k <= end_frame and v[1] is not None}
        err, kalman_estimated = process_data(data_points, solver)
        print(f'{solver}: {err}, len: {len(compute)}')

    # import pickle
    # compute = pickle.load(open(f"{data_dir}/gps_plot_data.pkl", "rb"))
    # err, kalman_estimated = process_data_2(compute[0], list(range(len(compute[0]))), 'a0')
    # err, kalman_estimated = process_data_2(compute[1], list(range(len(compute[1]))), 'a1')
    # err, kalman_estimated = process_data_2(compute[2], list(range(len(compute[2]))), 'a2')
