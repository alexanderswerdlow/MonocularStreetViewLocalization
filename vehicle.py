from collections import defaultdict
import os
import time
import pandas as pd
import numpy as np

from localization.feature_matching import extract_features, match_frame_features_to_panoramas, find_homography
from localization.segmentation import SemanticSegmentation
from download.query import query
from config import start_frame, recording_dir, scaled_frame_width, scaled_frame_height, SCALE_FACTOR, FRAME_WIDTH, data_dir
import cv2
from utilities import is_cv_cuda
import pickle
from localization.kvld import get_kvld_matches

class Vehicle:
    def __init__(self):
        self.log = pd.read_pickle(os.path.join(recording_dir, 'log.dat'))
        self.video = cv2.VideoCapture(os.path.join(recording_dir, 'Frames.m4v'))
        self.video.set(cv2.CAP_PROP_POS_FRAMES, start_frame-1)
        self.segmentation = SemanticSegmentation()
        self.frame_idx = 0

        self.saved_matches = pickle.load(open(f"{data_dir}/matches.p", "rb"))
        # TODO: Uncomment to save matches to file
        # self.saved_matches = defaultdict(list)

    def iterate_frames(self):
        start_row = self.log.index[(self.log['frame_number'] == start_frame + 2490) & (self.log['new_frame'] == 1)].tolist()[0]
        for _, row in self.log.iloc[start_row:].iterrows():
            if row['new_frame'] == 1:
                start_time = time.time()
                if is_cv_cuda():
                    frame = cv2.imread('0-frame.jpg')  # TODO: Fix reading frames w/CUDA
                else:
                    _, frame = self.video.read()
                self.localize_frame(frame, row)
                print(f'Frame {self.frame_idx} took: {time.time() - start_time}')
                self.frame_idx += 1

    def localize_frame(self, frame, metadata):
        self.match_frame_to_panorama(frame, metadata)
        # Do Visual Odometry

    def match_frame_to_panorama(self, frame, metadata):
        # TODO: Uncomment to save matches to file
        # panoramas = self.get_nearby_panoramas(metadata)
        # pano_data = self.extract_rectilinear_views(panoramas, metadata)
        # frame_data = self.process_frame(frame)
        
        # TODO: Uncomment to save matches to file
        # kvld_matches = get_kvld_matches((self.frame_idx, frame_data[0]), list(map(lambda x: (x[0].pano_id, x[1]), pano_data)))
        # for m in kvld_matches:
        #     points1, points2, desc = m[3], m[4], pano_data[0][-1]
        #     self.saved_matches[self.frame_idx].append((points1, points2, desc))
        # pickle.dump(self.kvld_matches, open(f"{data_dir}/matches.p", "wb"))
        
        if self.frame_idx in self.saved_matches:
            for points1, points2, desc in self.saved_matches[self.frame_idx]:
                print(find_homography(points1, points2, desc))
        else:
            print(f'Frame {self.frame_idx} not saved, exiting')
            exit()

        # matches = match_frame_features_to_panoramas(pano_data, frame_data)

    def process_frame(self, frame):
        frame = cv2.resize(frame, (scaled_frame_width, scaled_frame_height), interpolation=cv2.INTER_AREA)
        # frame = self.segmentation.segmentImage(frame)
        return frame, *extract_features(frame)

    def extract_rectilinear_views(self, panoramas, metadata, pitch=12):
        pano_data = []
        fov = np.rad2deg(np.arctan(FRAME_WIDTH/metadata['focal_length_x']))
        camera_matrix = np.array([
            [metadata['focal_length_x'], 0, metadata['principal_point_x']],
            [0, metadata['focal_length_y'], metadata['principal_point_y']],
            [0, 0, 0]
        ])/SCALE_FACTOR
        heading = metadata['course']
        for pano in panoramas:
            pano_data.append([pano, pano.get_rectilinear_image(heading, pitch, fov, scaled_frame_width, scaled_frame_height), camera_matrix])
        return pano_data

    def get_nearby_panoramas(self, metadata):
        loc = (metadata['latitude'], metadata['longitude'])
        return query(loc, n_points=10)

    # TODO: Integrate into visual odometry or delete
    def localize_two_frames(self, last_frame, frame):
        frame = cv2.resize(frame, (scaled_frame_width, scaled_frame_height), interpolation=cv2.INTER_AREA)
        last_frame = cv2.resize(last_frame, (scaled_frame_width, scaled_frame_height), interpolation=cv2.INTER_AREA)
        kp1, des1 = self.feature_tracker.extract_features(last_frame, save_features=True)
        kp2, des2 = self.feature_tracker.extract_features(frame, save_features=False)
        points1, points2, goodMatches = self.feature_tracker.match_features(kp2, des2)
        reference_img = cv2.drawMatchesKnn(last_frame, self.feature_tracker.current_frame_features[0], frame, kp2, goodMatches, None, flags=2)
        cv2.imshow('FLANN matched features', reference_img)
        cv2.waitKey(0)
