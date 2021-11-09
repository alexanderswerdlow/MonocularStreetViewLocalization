import cv2
import os
import numpy as np
import imutils

from utilities import convert_tuple_to_keypoints, load_pano_features
from config import images_dir, segmentation_model_dir

class FeatureTracker:
    def __init__(self):
        self.current_frame_features = None
        self.current_frame = None
        self.enet_neural_network = cv2.dnn.readNetFromTorch(os.path.join(segmentation_model_dir, 'enet-model.net'))
        self.class_names = (open(os.path.join(segmentation_model_dir, 'enet-classes.txt')).read().strip().split("\n"))

        self.IMG_COLOR_LIST = (
            open(os.path.join(segmentation_model_dir, 'enet-colors.txt')).read().strip().split("\n"))
        self.IMG_COLOR_LIST = [np.array(color.split(",")).astype(
            "int") for color in self.IMG_COLOR_LIST]
        self.IMG_COLOR_LIST = np.array(self.IMG_COLOR_LIST, dtype="uint8")

    def segmentImage(self, frame):
        RESIZED_WIDTH = 600
        IMG_NORM_RATIO = 1 / 255.0
        ENET_DIMENSIONS = (1024, 512)
        
        input_img = imutils.resize(frame, width=RESIZED_WIDTH)
        input_img_blob = cv2.dnn.blobFromImage(frame, IMG_NORM_RATIO, ENET_DIMENSIONS, 0, swapRB=True, crop=False)
        self.enet_neural_network.setInput(input_img_blob)
        enet_neural_network_output = self.enet_neural_network.forward()

        (number_of_classes, height, width) = enet_neural_network_output.shape[1:4] 
        class_map = np.argmax(enet_neural_network_output[0], axis=0)
        

        class_map_mask = cv2.resize(class_map, (
            input_img.shape[1], input_img.shape[0]),
                interpolation=cv2.INTER_NEAREST)

        # enet_neural_network_output = ((0.61 * class_map_mask) + (
        #     0.39 * input_img)).astype("uint8")

        class_legend = np.zeros(((len(self.class_names) * 25) + 25, 300, 3), dtype="uint8")
     
        # Put the class labels and colors on the legend
        for (i, (cl_name, cl_color)) in enumerate(zip(self.class_names, self.IMG_COLOR_LIST)):
            color_information = [int(color) for color in cl_color]
            cv2.putText(class_legend, cl_name, (5, (i * 25) + 17),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            cv2.rectangle(class_legend, (100, (i * 25)), (300, (i * 25) + 25),
                            tuple(color_information), -1)

        filtered_img = np.zeros_like(input_img)
        filtered_img[class_map_mask <= 5] = input_img[class_map_mask <= 5]
        # combined_images = np.concatenate((input_img, enet_neural_network_output), axis=0)
        cv2.imshow('Results', filtered_img)
        cv2.imshow("Class Legend", class_legend)
        cv2.waitKey(0)

    def extract_features(self, frame, show_keypoints=False):
        self.current_frame = frame
        self.segmentImage(frame)
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame_gray_eq = cv2.equalizeHist(frame_gray)
        detector = cv2.SIFT_create()
        # detector = cv2.SIFT_create(contrastThreshold=0.025)
        kp, des = detector.detectAndCompute(frame_gray_eq, None)
        des = np.float32(des)
        self.current_frame_features = (kp, des)

        if show_keypoints:
            img_kp = cv2.drawKeypoints(frame, kp, frame, color=(255,0,0))
            cv2.imshow('Keypoints', img_kp)
            cv2.waitKey(0)

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
        FLANN_INDEX_LSH = 6
        indexParams = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        # indexParams = dict(algorithm=FLANN_INDEX_LSH, table_number=6, key_size=12, multi_probe_level=1)
        searchParams = dict(checks=50)
        flann = cv2.FlannBasedMatcher(indexParams, searchParams)
        
        matches = flann.knnMatch(self.current_frame_features[1], reference_des, k=2)

        # Apply Lowe's ratio test
        goodMatches = []
        points1 = []
        points2 = []
        
        if len(matches) == 0:
            return points1, points2, goodMatches

        for m, n in matches:
            if m.distance < 0.7 * n.distance:
                goodMatches.append([m])
                points1.append(self.current_frame_features[0][m.queryIdx].pt)
                points2.append(reference_kp[n.trainIdx].pt)

        if reference_frame is not None:
            img = cv2.drawMatchesKnn(self.current_frame, self.current_frame_features[0], reference_frame, reference_kp, goodMatches, None, flags=2)
            img = cv2.drawMatchesKnn(cv2.equalizeHist(self.current_frame), self.current_frame_features[0], cv2.equalizeHist(reference_frame), reference_kp, goodMatches, None, flags=2)
            cv2.imshow('FLANN matched features', img)
            cv2.waitKey(0)

        return points1, points2, goodMatches

    def find_homography(self, points1, points2, K):
        E, mask = cv2.findEssentialMat(points1, points2, cameraMatrix=K, method=cv2.RANSAC)
        R, t = cv2.recoverPose(E, points1, points2, K, mask=mask)