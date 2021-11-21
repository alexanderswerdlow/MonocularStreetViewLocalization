from collections import defaultdict
import os
import time
import pandas as pd
import numpy as np

from localization.segmentation import SemanticSegmentation
from download.query import query
from config import start_frame, recording_dir, scaled_frame_width, scaled_frame_height, SCALE_FACTOR, FRAME_WIDTH, data_dir
import cv2
from utilities import is_cv_cuda
import pickle
from localization.kvld import get_kvld_matches
import copyreg


def _pickle_dmatch(dmatch):
    return cv2.DMatch, (dmatch.queryIdx, dmatch.trainIdx, dmatch.imgIdx, dmatch.distance)


copyreg.pickle(cv2.DMatch().__class__, _pickle_dmatch)


class Vehicle:
    def __init__(self):
        self.log = pd.read_pickle(os.path.join(recording_dir, 'log.dat'))
        self.video = cv2.VideoCapture(os.path.join(recording_dir, 'Frames.m4v'))
        self.video.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        self.segmentation = SemanticSegmentation()
        self.frame_idx = start_frame

        try:
            self.saved_matches = pickle.load(open(f"{data_dir}/kvld_matches.p", "rb"))
        except (OSError, IOError) as e:
            self.saved_matches = {}

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
        # if self.frame_idx in self.saved_matches:
        #     breakpoint()

        if self.frame_idx % 10 == 0 and self.frame_idx not in self.saved_matches:
            print(f'Starting on Frame: {self.frame_idx}')
            panoramas = self.get_nearby_panoramas(metadata)
            pano_data = self.extract_rectilinear_views(panoramas, metadata)
            frame_data = self.process_frame(frame)

            pano_dict = {p[0].pano_id: p for p in pano_data}
            kvld_matches = get_kvld_matches((self.frame_idx, frame_data), pano_dict)
            self.saved_matches[self.frame_idx] = (kvld_matches, metadata)
        #    pickle.dump(self.saved_matches, open(f"{data_dir}/kvld_matches.p", "wb"))


    def process_frame(self, frame):
        frame = cv2.resize(frame, (scaled_frame_width, scaled_frame_height), interpolation=cv2.INTER_AREA)
        # frame = self.segmentation.segmentImage(frame)
        return frame

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
