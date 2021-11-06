import cv2

class FeatureTracker:
    def __init__(self, stream, debug=False):
        self.stream = stream
        self.debug = debug

    def extract_features(self, frame_gray):
        detector = cv2.SIFT()
        kp, des = detector.detectAndCompute(frame_gray, None)
        if self.debug:
            img = cv2.drawKeypoints(frame_gray, kp)
            cv2.imshow('Keypoints', img)
            cv2.waitKey(0)

    def match_features(self, kp1, des1, frame1, kp2, des2, frame2):
        # Create FLANN matcher object
        FLANN_INDEX_KDTREE = 0
        indexParams = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        searchParams = dict(checks=50)
        flann = cv2.FlannBasedMatcher(indexParams, searchParams)

        matches = flann.knnMatch(des1, des2, k=2)

        # Apply Lowe's ratio test
        goodMatches = []
        points1 = []
        points2 = []
        

        for m, n in matches:
            if m.distance < 0.75 * n.distance:
                goodMatches.append([m])
                points1.append(kp1[m.trainIdx].pt)
                points2.append(kp2[n.trainIdx].pt)

        if self.debug:
            img = cv2.drawMatchesKnn(frame1, kp1, frame2, kp2, goodMatches, None, flags=2)
            cv2.imshow('FLANN matched features', img)
            cv2.waitKey(0)

        return points1, points2, goodMatches

    def find_homography(self, points1, points2, K):
        E, mask = cv2.findEssentialMat(points1, points2, cameraMatrix=K, method=cv2.RANSAC)
        R, t = cv2.recoverPose(E, points1, points2, K, mask=mask)