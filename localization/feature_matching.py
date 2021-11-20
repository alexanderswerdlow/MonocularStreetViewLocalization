import cv2
import numpy as np
import multiprocessing
import time
from functools import partial
import itertools
from utilities import is_cv_cuda

if is_cv_cuda():
    surf_gpu = cv2.cuda.SURF_CUDA_create(400)
    matcher_gpu = cv2.cuda.DescriptorMatcher_createBFMatcher(cv2.NORM_L2)

def extract_features(frame, show_keypoints=False):
    if is_cv_cuda():
        gpu_grey = cv2.cuda.cvtColor(cv2.cuda_GpuMat(frame), cv2.COLOR_RGB2GRAY)
        gpu_grey_eq = cv2.cuda.equalizeHist(gpu_grey)
        kp, des = surf_gpu.detectWithDescriptors(gpu_grey_eq, None)
    else:
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame_gray_eq = cv2.equalizeHist(frame_gray)
        detector = cv2.SIFT_create()
        # detector = cv2.SIFT_create(contrastThreshold=0.025)
        kp, des = detector.detectAndCompute(frame_gray_eq, None)
        des = np.float32(des)

    if show_keypoints:
        img_kp = cv2.drawKeypoints(frame.copy(), kp, frame, color=(255, 0, 0))
        cv2.imwrite(f'tmp/keypoints-{time.time_ns() - 1636597296826147000}.jpg', img_kp)

    return (kp, des)


def match_frame_features_to_panoramas_slow(panoramas, frame_data):
    with multiprocessing.Pool(5) as p:
        matches = list(itertools.chain.from_iterable(p.map(partial(match_frame_features_to_single_panorama, frame_data=frame_data), panoramas)))
        matches.sort(key=lambda row: row[-1], reverse=True)
        return matches


def match_frame_features_to_panoramas(pano_data, frame_data):
    matches = [match_frame_features_to_single_panorama(p, frame_data) for p in pano_data]
    matches = [item for sublist in matches for item in sublist]
    matches.sort(key=lambda row: row[-1], reverse=True)
    return matches


def match_frame_features_to_single_panorama(pano_data, frame_data):
    pano, rectilinear, cam_mtx = pano_data
    matches = []

    if is_cv_cuda():
        points1, points2, goodMatches = match_features_cuda((rectilinear, None, None), frame_data)
        matches.append([pano, rectilinear, None, points1, points2, goodMatches, len(goodMatches)])
    else:
        kp, des = extract_features(rectilinear)
        des = np.float32(des)
        points1, points2, goodMatches = match_features((None, kp, des), frame_data)
        matches.append([pano, rectilinear, kp, points1, points2, goodMatches, len(goodMatches)])

    return matches


def match_features(reference_data, frame_data):
    reference_frame, reference_kp, reference_des = reference_data
    frame_frame, frame_kp, frame_des = frame_data
    # Create FLANN matcher object
    FLANN_INDEX_KDTREE = 0
    FLANN_INDEX_LSH = 6
    indexParams = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    # indexParams = dict(algorithm=FLANN_INDEX_LSH, table_number=6, key_size=12, multi_probe_level=1)
    searchParams = dict(checks=50)
    flann = cv2.FlannBasedMatcher(indexParams, searchParams)

    matches = flann.knnMatch(frame_des, reference_des, k=2)

    # Apply Lowe's ratio test
    goodMatches = []
    points1 = []
    points2 = []

    if len(matches) == 0:
        return points1, points2, goodMatches

    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            goodMatches.append([m])
            points1.append(frame_kp[m.queryIdx].pt)
            points2.append(reference_kp[n.trainIdx].pt)

    if reference_frame is not None:
        img = cv2.drawMatchesKnn(frame_frame, frame_kp, reference_frame, reference_kp, goodMatches, None, flags=2)
        img = cv2.drawMatchesKnn(cv2.equalizeHist(frame_frame), frame_kp, cv2.equalizeHist(reference_frame), reference_kp, goodMatches, None, flags=2)
        cv2.imshow('FLANN matched features', img)
        cv2.waitKey(0)

    return points1, points2, goodMatches


def match_features_cuda(reference_data, frame_data):
    reference_frame, reference_kp, reference_des = reference_data
    frame_frame, frame_kp, frame_des = frame_data

    reference_frame_gpu_grey = cv2.cuda.cvtColor(cv2.cuda_GpuMat(reference_frame), cv2.COLOR_BGR2GRAY)
    reference_kp_gpu, reference_des_gpu = surf_gpu.detectWithDescriptors(reference_frame_gpu_grey, None)
    matches = matcher_gpu.knnMatch(reference_des_gpu, frame_des, k=2)
    kp1 = cv2.cuda_SURF_CUDA.downloadKeypoints(surf_gpu, reference_kp_gpu)
    kp2 = cv2.cuda_SURF_CUDA.downloadKeypoints(surf_gpu, frame_kp)

    # Apply Lowe's ratio test
    goodMatches = []
    points1 = []
    points2 = []

    if len(matches) == 0:
        return points1, points2, goodMatches

    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            goodMatches.append([m])
            points1.append(kp1[m.queryIdx].pt)
            points2.append(kp2[n.trainIdx].pt)

    return points1, points2, goodMatches
