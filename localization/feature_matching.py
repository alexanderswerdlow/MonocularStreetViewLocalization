import cv2
import numpy as np

def find_homography(points1, points2, K):
    points1, points2 = np.array(points1), np.array(points2)
    E, mask = cv2.findEssentialMat(points1, points2, cameraMatrix=K, method=cv2.RANSAC)
    points, R, t, mask = cv2.recoverPose(E, points1, points2, K, mask=mask)
    return R, t
