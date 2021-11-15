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
from config import images_dir, start_frame, headings_
from itertools import islice

import copyreg
import cv2


def _pickle_keypoint(keypoint):
    return cv2.KeyPoint, (
        keypoint.pt[0],
        keypoint.pt[1],
        keypoint.size,
        keypoint.angle,
        keypoint.response,
        keypoint.octave,
        keypoint.class_id,
    )


copyreg.pickle(cv2.KeyPoint().__class__, _pickle_keypoint)


class LocalizationProcess(multiprocessing.Process):
    def __init__(self, data_queue, location_queue):
        multiprocessing.Process.__init__(self)
        self.data_queue = data_queue
        self.location_queue = location_queue
        self.previous_localized_position = None
        self.previous_frame_features = None

    def run(self):
        print(f"Starting Localization Process: {time.time()}")
        last_frame = None
        for feature_tracker, _, loc, heading in iter(self.data_queue.get, None):
            print('Get frame')
            if feature_tracker is None:
                print("Exiting Localization Process")
                self.data_queue.task_done()
                break

            self.localize_two_panoramas(feature_tracker, loc, heading)
            print('Localized Pano')
            # if last_frame is not None:
            #    self.localize_two_frames(last_frame, frame)
            #last_frame = frame
            # location = None # replace
            self.location_queue.put(loc)
            self.data_queue.task_done()
            print("Task Done")

    def localize_two_panoramas(self, feature_tracker: FeatureTracker, loc: Loc, heading):
        print('Starting localize...Querying')
        panos_to_view = (headings_[headings_ > heading].min(), headings_[headings_ < heading].max())
        panoramas = [(p, panos_to_view) for p in query(loc, n_points=10)]

        # for p, (l, h) in panoramas:
        #     cv2.imwrite(f'tmp/{p.pano_id}-{l}.jpg', cv2.imread(f'{images_dir}/{p.pano_id}-{l}.jpg'))
        #     cv2.imwrite(f'tmp/{p.pano_id}-{h}.jpg', cv2.imread(f'{images_dir}/{p.pano_id}-{h}.jpg'))

        print('Matching with panoramas...')
        matches = feature_tracker.find_best_pano_images(panoramas)
        print(f'Found best pano matches: {len(matches)}')
        
        for _, match in enumerate(islice(matches, 0, 50)):
            print(f'Match with number of features: {match[-1]}')
            reference_img = cv2.imread(os.path.join(images_dir, f'{match[0].pano_id}-{match[1]}.jpg'))
            reference_img = cv2.drawMatchesKnn(feature_tracker.current_frame, feature_tracker.current_frame_features[0], reference_img, match[2], match[5], None, flags=2)
            cv2.imwrite(f'tmp/flann-match-{time.time_ns() - 1636597296826147000}.jpg', reference_img)

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

        matches.sort(key=lambda row: row[-1])
        print(matches)
        self.previous_frame_features = self.feature_tracker.current_frame_features

    def localize_two_frames(self, last_frame, frame):
        frame = cv2.resize(frame, (640, int(640*frame.shape[0]/frame.shape[1])), interpolation=cv2.INTER_AREA)
        last_frame = cv2.resize(last_frame, (640, int(640*last_frame.shape[0]/last_frame.shape[1])), interpolation=cv2.INTER_AREA)
        kp1, des1 = self.feature_tracker.extract_features(last_frame, save_features=True)
        kp2, des2 = self.feature_tracker.extract_features(frame, save_features=False)
        points1, points2, goodMatches = self.feature_tracker.match_features(kp2, des2)
        reference_img = cv2.drawMatchesKnn(last_frame, self.feature_tracker.current_frame_features[0], frame, kp2, goodMatches, None, flags=2)
        cv2.imshow('FLANN matched features', reference_img)
        cv2.waitKey(0)


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
        print(f"Vehicle Init Localization: {time.time()}")
        self.localization_process.start()
        print("Vehicle Init Stream")
        self.stream_log(stream_freq)

        print("Waiting to Join Localization Process")
        self.data_queue.join()
        self.localization_process.join()
        print("Terminated Localization Process")

    def stream_log(self, stream_freq):
        time.sleep(1)
        self.video.set(cv2.CAP_PROP_POS_FRAMES, start_frame-1)
        start_row = self.log.index[(self.log['frame_number'] == start_frame + 2490) & (self.log['new_frame'] == 1)].tolist()[0]
        frame_idx = 0
        for _, row in self.log.iloc[start_row:].iterrows():
            if self.exit_thread:
                return

            # read other sensor data here
            if row['new_frame'] == 1:
                _, frame = self.video.read()
                # if frame_idx < 2:
                #     cv2.imwrite(f'tmp/{frame_idx}_pano.jpg', frame)
                frame = cv2.resize(frame, (640, int(640*frame.shape[0]/frame.shape[1])), interpolation=cv2.INTER_AREA)
                feature_tracker = FeatureTracker()
                feature_tracker.extract_features(frame, save_features=True)
                self.data_queue.put((feature_tracker, frame_idx, Loc(row['latitude'], row['longitude']), row['course']))
                print(f'Put frame: {frame_idx}')
                frame_idx += 1

                time.sleep(1.0)

    def close(self):
        self.exit_thread = True
        self.localization_process.kill()
