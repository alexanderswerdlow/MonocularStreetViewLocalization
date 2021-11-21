from collections import defaultdict
import os
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from localization.segmentation import SemanticSegmentation
from download.query import query
from config import start_frame, images_dir, recording_dir, scaled_frame_width, scaled_frame_height, SCALE_FACTOR, FRAME_WIDTH, data_dir
import cv2
from utilities import is_cv_cuda
import pickle
from localization.kvld import get_kvld_matches
from localization.localization import find_homography, show_rgbd, estimate_location_two_panoramas
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
        # fov = np.rad2deg(np.arctan(FRAME_WIDTH/metadata['focal_length_x']))
        # locations = []
        # directions = []
        # n = 2
        # num_matches = []
        # if self.frame_idx in self.saved_matches:
        #     for points1, points2, camera_matrix, pano in self.saved_matches[self.frame_idx]:
        #         # rect = pano.get_rectilinear_depth(metadata['course'], 12, fov)
        #         # rgb = cv2.imread(f'{images_dir}/{pano.pano_id}.jpg')
        #         # show_rgbd(rgb, rect)
        #         num_matches.append(len(points1))
        #         locations.append([pano.lat, pano.long])
        #         R, t = find_homography(points1, points2, camera_matrix)
        #         directions.append([t[0], t[2]])
        # else:
        #     print(f'Frame {self.frame_idx} not saved, exiting')
        #     exit()
        # i = np.argpartition(num_matches, -n)[-n:]
        # # plot(np.array(locations)[i], np.array(directions)[i])

        # localized_point = estimate_location_two_panoramas(np.array(locations)[i], np.array(directions)[i], metadata['course']) 
        # print(localized_point)
        # # matches = match_frame_features_to_panoramas(pano_data, frame_data)
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

    def extract_rectilinear_views(self, panoramas, metadata, fov, pitch=12):
        pano_data = []
        heading = metadata['course']
        for pano in panoramas:
            pano_data.append([pano, pano.get_rectilinear_image(heading, pitch, fov, scaled_frame_width, scaled_frame_height), metadata['camera_matrix']])
        return pano_data

    def get_nearby_panoramas(self, metadata):
        loc = (metadata['latitude'], metadata['longitude'])
        return query(loc, n_points=10)
