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
    gmap3.draw(f"{data_dir}/{solver}-kalman.html")

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
    min_errors = calculate_error(trajectory, estimated)
    kalman_estimated = kalman_filter(estimated)
    min_errors_kalman = calculate_error(trajectory, kalman_estimated)
    plot_stuff(min_errors, indices, solver)
    plot_stuff(min_errors_kalman, indices, solver + '-kalman')
    save_map(kalman_estimated, trajectory, solver)
    # except:
    #     breakpoint()
    return np.sqrt(np.mean(min_errors**2)), kalman_estimated


if __name__ == '__main__':
    import pickle
    compute = pickle.load(open(f"{data_dir}/compute_good.p", "rb"))  # compute_good
    from download.waypoints import reference
    trajectory = np.array(reference)

    estimated = []
    errors = []
    for l, (estimate, error, localized_coord) in compute.items():
        # print(estimate.cost, error)
        if 5000 < l < 9000:  # estimate.cost < 300e16 and
            estimated.append((localized_coord.latitude, localized_coord.longitude))
            errors.append(error)

    estimated = np.array(estimated)

    print(f'MSE: {np.sqrt(np.mean(dist**2))}, {len(estimated)}')

    import matplotlib.ticker as mticker

    x = list(range(0, len(estimated)))
    # plt.gca().xaxis.set_major_locator(mticker.MultipleLocator(1))
    plt.title("Distance to Ground Truth")
    plt.xlabel("Estimated Point")
    plt.ylabel("Min Distance to Reference Trajectory (Meters)")
    plt.plot(x, dist)
    plt.savefig('error.png', bbox_inches='tight', dpi=600)

    plt.clf()
    x = list(range(0, len(estimated)))
    # plt.gca().xaxis.set_major_locator(mticker.MultipleLocator(1))
    plt.title("Mean Pixel Re-Projection Error Per Frame")
    plt.xlabel("Estimated Point")
    plt.ylabel("Mean Euclidean Distance Error (Pixels)")
    plt.plot(x, errors)
    plt.savefig('reproj.png', bbox_inches='tight', dpi=600)

    from localization.localization import CustomGoogleMapPlotter
    localized_coord = geopy.distance.distance(meters=mag).destination(locations[0], bearing=np.rad2deg(bearing))
    gmap3 = CustomGoogleMapPlotter(34.060458, -118.437621, 17, apikey=api_key)
    gmap3.scatter(trajectory[:,0], trajectory[:,1], '#0000FF', size=5, marker=True)
    gmap3.scatter(estimated[:,0], estimated[:,1], '#FF0000', size=5, marker=True)
    gmap3.scatter([localized_coord.latitude], [localized_coord.longitude], '#0000FF', size=7, marker=True)
    gmap3.draw(f"{data_dir}/trajectory_offsets.html")
