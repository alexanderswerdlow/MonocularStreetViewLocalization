import cv2
import numpy as np
import multiprocessing
import time
from utilities import convert_tuple_to_keypoints, load_pano_features
from functools import partial
import itertools


def extract_features(frame, show_keypoints=False):
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
    with multiprocessing.Pool(system_cores) as p:
        matches = list(itertools.chain.from_iterable(p.map(partial(match_frame_features_to_single_panorama, frame_data=frame_data), panoramas)))
        matches.sort(key=lambda row: row[-1], reverse=True)
        return matches


def match_frame_features_to_panoramas(panoramas, frame_data):
    matches = [match_frame_features_to_single_panorama(p, frame_data) for p in panoramas]
    matches = [item for sublist in matches for item in sublist]
    matches.sort(key=lambda row: row[-1], reverse=True)
    return matches


def match_frame_features_to_single_panorama(pano_data, frame_data):
    pano, headings = pano_data
    # print(f'Loading pano features...{pano.pano_id}')
    features_dict = load_pano_features(pano.pano_id)
    matches = []
    for heading in headings:
        # print(f'Matching feature: {pano.pano_id},{heading}')
        kp, des = features_dict[heading]
        kp = convert_tuple_to_keypoints(kp)
        des = np.float32(des)
        points1, points2, goodMatches = match_features((None, kp, des), frame_data)
        matches.append([pano, heading, kp, points1, points2, goodMatches, len(goodMatches)])

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


def find_homography(self, points1, points2, K):
    E, mask = cv2.findEssentialMat(points1, points2, cameraMatrix=K, method=cv2.RANSAC)
    R, t = cv2.recoverPose(E, points1, points2, K, mask=mask)
