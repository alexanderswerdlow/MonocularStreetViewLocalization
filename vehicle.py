import multiprocessing
import threading

import os
import time
import cv2
import pandas as pd
import numpy as np

from localization.feature_matching import FeatureTracker
from localization.segmentation import SemanticSegmentation
from download.query import query
from download.util import Loc
from utilities import convert_tuple_to_keypoints, load_pano_features
from config import images_dir, skip_frames

class LocalizationProcess(multiprocessing.Process):
    def __init__(self, data_queue, location_queue):
        multiprocessing.Process.__init__(self)
        self.data_queue = data_queue
        self.location_queue = location_queue
        self.feature_tracker = FeatureTracker()
        self.previous_localized_position = None
        self.previous_frame_features = None

        self.frame_count = 0
    
    def run(self):
        print("Starting Localization Process")
        while True:
            frame, gps = self.data_queue.get() #blocking
            if frame is None:
                print("Exiting Localization Process")
                self.data_queue.task_done()
                break

            self.frame_count += 1
            if self.frame_count < skip_frames:
                continue

            # localize frame
            self.localize_two_panoramas(frame, gps)
            location = None # replace
            self.location_queue.put(location)
            self.data_queue.task_done()

    def localize_two_panoramas(self, frame, location):
        frame = cv2.resize(frame, (640, int(640*frame.shape[0]/frame.shape[1])), interpolation=cv2.INTER_AREA)
        # loc = Loc(location[0], location[1])
        loc = Loc(34.060644, -118.445362)
        panoramas = query(loc, n_points=10, distance_upper_bound=100)
        print([p.pano_id for p in panoramas])
        self.feature_tracker.extract_features(frame, show_keypoints=True)
        matches = self.feature_tracker.find_best_pano_images(panoramas)
        n_matches = 30
        cv2.imshow('frame', frame)
        cv2.waitKey(0)
        for i in range(n_matches):
            match = matches[i]
            print(f'Match with number of features: {match[-1]}')
            reference_img = cv2.imread(os.path.join(images_dir, f'{match[0].pano_id}-{match[1]}.jpg'))
            reference_img = cv2.drawMatchesKnn(frame, self.feature_tracker.current_frame_features[0], reference_img, match[2], match[-2], None, flags=2)
            cv2.imshow('FLANN matched features', reference_img)
            cv2.waitKey(0)
            

    def localize_panorama_prev_frame(self, frame):
        self.feature_tracker.extract_features(frame)
        loc = Loc(self.previous_localized_position[0], self.previous_localized_position[1])
        panoramas = query(loc, n_points=1, distance_upper_bound=50)
        matches = []
        for pano in panoramas:
            features_dict = load_pano_features(pano.pano_id)
            for heading, features in features_dict.items():
                kp, des = features
                kp = convert_tuple_to_keypoints(kp)
                points1, points2, goodMatches = self.feature_tracker.match_features(kp, des, cv2.imread(os.path.join(images_dir, f'{pano.pano_id}-{heading}.jpg')))
                matches.append([pano, heading, points1, points2, len(goodMatches)])
                print(len(goodMatches))
        matches.sort(key=lambda row: row[-1])
        print(matches)
        self.previous_frame_features = self.feature_tracker.current_frame_features


class Vehicle:
    def __init__(self, data_dir):
        self.queried_panoramas = []
        self.data_queue = multiprocessing.JoinableQueue()
        self.location_queue = multiprocessing.Queue()

        self.log = pd.read_pickle(os.path.join(data_dir, 'log.dat'))
        self.video = cv2.VideoCapture(os.path.join(data_dir, 'Frames.m4v'))

        self.exit_thread = False

    def start(self, stream_freq):
        self.localization_process = LocalizationProcess(self.data_queue, self.location_queue)
        self.localization_process.start()
        self.stream_log(stream_freq)

        # print("Waiting to Join Localization Process")
        # self.data_queue.join()
        # self.localization_process.join()
        # print("Terminated Localization Process")

    def stream_log(self, frequency):
        for index, row in self.log.iterrows():
            if self.exit_thread:
                return

            gps = [row['latitude'], row['longitude']]
            # read other sensor data here
            if row['new_frame'] == 1:
                ret, frame = self.video.read()
                self.data_queue.put([frame, gps])
            time.sleep(1/frequency)

    def close(self, sig, frame):
        self.exit_thread = True
        self.localization_process.kill()