import os
import time
import pandas as pd
import numpy as np

from localization.feature_matching import extract_features, match_frame_features_to_panoramas, find_homography
from localization.segmentation import SemanticSegmentation
from download.query import query
from config import images_dir, start_frame, recording_dir
from itertools import islice
import cv2
from utilities import is_cv_cuda

FRAME_WIDTH = 640

class Vehicle:
    def __init__(self):
        self.log = pd.read_pickle(os.path.join(recording_dir, 'log.dat'))
        self.video = cv2.VideoCapture(os.path.join(recording_dir, 'Frames.m4v'))
        self.video.set(cv2.CAP_PROP_POS_FRAMES, start_frame-1)
        self.segmentation = SemanticSegmentation()
        self.counter = 0

    def iterate_frames(self):
        start_row = self.log.index[(self.log['frame_number'] == start_frame + 2490) & (self.log['new_frame'] == 1)].tolist()[0]
        frame_idx = 0
        for _, row in self.log.iloc[start_row:].iterrows():
            if row['new_frame'] == 1:
                start_time = time.time()
                if is_cv_cuda():
                    frame = cv2.imread('0-frame.jpg') # TODO: Fix reading frames w/CUDA
                else:
                    _, frame = self.video.read()
                self.localize_frame(frame, row)
                print(f'Frame {frame_idx} took: {time.time() - start_time}')
                frame_idx += 1

    def localize_frame(self, frame, metadata):
        self.match_frame_to_panorama(frame, metadata)
        # Do Visual Odometry

    def match_frame_to_panorama(self, frame, metadata):
        panoramas = self.get_nearby_panoramas(metadata)

        self.counter += 1

        pano_data = self.extract_rectilinear_views(panoramas, metadata['focal_length_x'], metadata['course'])
        frame_data = self.process_frame(frame)
        from localization.kvld import get_kvld_matches
        kvld_matches = get_kvld_matches((self.counter, frame_data[0]), list(map(lambda x: (x[0].pano_id, x[1]), pano_data)))
        print(find_homography(kvld_matches[0][3], kvld_matches[0][4], np.eye(3)))
        matches = match_frame_features_to_panoramas(pano_data, frame_data)
        # print(matches)

    def process_frame(self, frame):
        frame = cv2.resize(frame, (640, int(640*frame.shape[0]/frame.shape[1])), interpolation=cv2.INTER_AREA)
        # frame = self.segmentation.segmentImage(frame)
        return frame, *extract_features(frame)

    def extract_rectilinear_views(self, panoramas, focal_length, heading, pitch=10, fov=100, w=640, h=480):
        pano_data = []
        for pano in panoramas:
            pano_data.append([pano, pano.get_rectilinear_image(heading, pitch, np.rad2deg(np.arctan(1920/focal_length)), w, h)])
        return pano_data

    def get_nearby_panoramas(self, metadata):
        loc = (metadata['latitude'], metadata['longitude'])
        return query(loc, n_points=10)
        
    # TODO: Integrate into visual odometry or delete
    def localize_two_frames(self, last_frame, frame):
        frame = cv2.resize(frame, (640, int(640*frame.shape[0]/frame.shape[1])), interpolation=cv2.INTER_AREA)
        last_frame = cv2.resize(last_frame, (640, int(640*last_frame.shape[0]/last_frame.shape[1])), interpolation=cv2.INTER_AREA)
        kp1, des1 = self.feature_tracker.extract_features(last_frame, save_features=True)
        kp2, des2 = self.feature_tracker.extract_features(frame, save_features=False)
        points1, points2, goodMatches = self.feature_tracker.match_features(kp2, des2)
        reference_img = cv2.drawMatchesKnn(last_frame, self.feature_tracker.current_frame_features[0], frame, kp2, goodMatches, None, flags=2)
        cv2.imshow('FLANN matched features', reference_img)
        cv2.waitKey(0)
