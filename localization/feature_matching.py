import cv2
import os
import numpy as np

from utilities import convert_tuple_to_keypoints, load_pano_features
from config import images_dir

class FeatureTracker:
    def __init__(self):
        self.current_frame_features = None
        self.current_frame = None

    def extract_features(self, frame):
        self.current_frame = frame
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        detector = cv2.SIFT_create(contrastThreshold=0.025)
        kp, des = detector.detectAndCompute(frame_gray, None)
        self.current_frame_features = (kp, des)

    def find_best_pano_images(self, panoramas):
        matches = []
        for pano in panoramas:
            features_dict = load_pano_features(pano.pano_id)
            for heading, features in features_dict.items():
                kp, des = features
                kp = convert_tuple_to_keypoints(kp)
                des = np.float32(des)
                points1, points2, goodMatches = self.match_features(kp, des)
                matches.append([pano, heading, kp, points1, points2, goodMatches, len(goodMatches)])
                print(len(goodMatches))
        matches.sort(key=lambda row: row[-1], reverse=True)
        return matches

    def match_features(self, reference_kp, reference_des, reference_frame=None):
        # Create FLANN matcher object
        FLANN_INDEX_KDTREE = 0
        indexParams = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        searchParams = dict(checks=50)
        flann = cv2.FlannBasedMatcher(indexParams, searchParams)

        matches = flann.knnMatch(self.current_frame_features[1], reference_des, k=2)

        # Apply Lowe's ratio test
        goodMatches = []
        points1 = []
        points2 = []
        
        for m, n in matches:
            if m.distance < 0.7 * n.distance:
                goodMatches.append([m])
                points1.append(self.current_frame_features[0][m.trainIdx].pt)
                points2.append(reference_kp[n.trainIdx].pt)

        if reference_frame is not None:
            img = cv2.drawMatchesKnn(self.current_frame, self.current_frame_features[0], reference_frame, reference_kp, goodMatches, None, flags=2)
            cv2.imshow('FLANN matched features', img)
            cv2.waitKey(0)

        return points1, points2, goodMatches

    def find_homography(self, points1, points2, K):
        E, mask = cv2.findEssentialMat(points1, points2, cameraMatrix=K, method=cv2.RANSAC)
        R, t = cv2.recoverPose(E, points1, points2, K, mask=mask)