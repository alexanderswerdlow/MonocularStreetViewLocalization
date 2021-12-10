from collections import defaultdict
import os
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from localization.segmentation import SemanticSegmentation
from download.query import query
from config import images_dir, recording_dir, scaled_frame_width, scaled_frame_height, SCALE_FACTOR, FRAME_WIDTH, data_dir
import cv2
from utilities import is_cv_cuda
import pickle
from localization.kvld import get_kvld_matches
import copyreg


def _pickle_dmatch(dmatch):
    return cv2.DMatch, (dmatch.queryIdx, dmatch.trainIdx, dmatch.imgIdx, dmatch.distance)


copyreg.pickle(cv2.DMatch().__class__, _pickle_dmatch)


class Vehicle:
    def __init__(self, start_frame):
        self.start_frame = start_frame
        self.log = pd.read_pickle(os.path.join(recording_dir, 'log.dat'))
        self.video = cv2.VideoCapture(os.path.join(recording_dir, 'Frames.m4v'))
        self.video.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        self.segmentation = SemanticSegmentation()
        self.frame_idx = start_frame
        self.save_file = f"{data_dir}/kvld_matches_{self.start_frame}.p"

        try:
            self.saved_matches = pickle.load(open(self.save_file, "rb"))
        except (OSError, IOError) as e:
            self.saved_matches = {}

    def iterate_frames(self):
        start_row = self.log.index[(self.log['frame_number'] == self.start_frame + 2490) & (self.log['new_frame'] == 1)].tolist()[0]
        for _, row in self.log.iloc[start_row:].iterrows():
            if row['new_frame'] == 1:
                start_time = time.time()
                _, frame = self.video.read()
                row['camera_matrix'] = self.format_camera_matrix(row)
                self.localize_frame(frame, row)
                print(f'Frame {self.frame_idx} took: {time.time() - start_time}')
                self.frame_idx += 1

    def format_camera_matrix(self, metadata):
        camera_matrix = np.array([
            [metadata['focal_length_x'], 0, metadata['principal_point_x']],
            [0, metadata['focal_length_y'], metadata['principal_point_y']],
            [0, 0, 0]
        ])/SCALE_FACTOR
        return camera_matrix

    def localize_frame(self, frame, metadata):
        self.match_frame_to_panorama(frame, metadata)
        # Do Visual Odometry

    def match_frame_to_panorama(self, frame, metadata):
        fov = np.rad2deg(np.arctan(FRAME_WIDTH/metadata['focal_length_x']))

        if self.frame_idx not in self.saved_matches and self.frame_idx % 5 == 0:
            print(f'Starting on Frame: {self.frame_idx}')
            panoramas = self.get_nearby_panoramas(metadata)
            pano_data = self.extract_rectilinear_views(panoramas, metadata, fov)
            frame_data = self.process_frame(frame)

            pano_dict = {p[0].pano_id: p for p in pano_data}
            kvld_matches = get_kvld_matches((self.frame_idx, frame_data), pano_dict, self.start_frame)
            self.saved_matches[self.frame_idx] = (kvld_matches, metadata)
            pickle.dump(self.saved_matches, open(self.save_file, "wb"))
            
    def plot_pano_features_subset(self, panos, matches, pano_points):
        for i, (pano, im) in enumerate(panos):
            match = matches[i]
            image = im.copy()
            all_features = np.array(match[1]).astype(int)
            filtered_features = np.array(pano_points[i]).astype(int)
            for feature in all_features:
                cv2.circle(image, feature, 10, (0, 0, 255), -1)
            
            for feature in filtered_features:
                cv2.circle(image, feature, 20, (255, 0, 0), -1)
        
            cv2.imwrite(f'{data_dir}/testing/features_{i}.png', image)
            
    def get_angles(self, d, heading):
        d /= np.linalg.norm(d, axis=1)[:, np.newaxis]
        angles = heading - np.rad2deg(np.arctan2(d[:,1], d[:,0]))
        return angles

    def process_frame(self, frame):
        frame = cv2.resize(frame, (scaled_frame_width, scaled_frame_height), interpolation=cv2.INTER_AREA)
        # frame = self.segmentation.segmentImage(frame)
        return frame

    def extract_rectilinear_views(self, panoramas, metadata, fov, pitch=12):
        pano_data = []
        heading = metadata['course']
        for pano in panoramas:
            pano_data.append([pano, pano.get_rectilinear_image(heading, pitch, fov, scaled_frame_width, scaled_frame_height), metadata['camera_matrix']])
        return pano_data

    def get_nearby_panoramas(self, metadata):
        loc = (metadata['latitude'], metadata['longitude'])
        return query(loc, n_points=6)
