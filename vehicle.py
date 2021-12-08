from collections import defaultdict
import os
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from localization.segmentation import SemanticSegmentation
from download.query import query
from config import start_frame, images_dir, recording_dir, scaled_frame_width, scaled_frame_height, SCALE_FACTOR, FRAME_WIDTH, data_dir, api_key
import cv2
from utilities import is_cv_cuda
import pickle
from localization.kvld import get_kvld_matches
from localization.localization import *
import copyreg


def _pickle_dmatch(dmatch):
    return cv2.DMatch, (dmatch.queryIdx, dmatch.trainIdx, dmatch.imgIdx, dmatch.distance)


copyreg.pickle(cv2.DMatch().__class__, _pickle_dmatch)


class Vehicle:
    def __init__(self, solver):
        self.solver = solver
        self.log = pd.read_pickle(os.path.join(recording_dir, 'log.dat'))
        self.video = cv2.VideoCapture(os.path.join(recording_dir, 'Frames.m4v'))
        self.video.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        self.segmentation = SemanticSegmentation()
        self.frame_idx = start_frame
        self.gmap3 = CustomGoogleMapPlotter(34.060458, -118.437621, 17, apikey=api_key)
        self.frames_processed = 0

        try:
            self.saved_matches = pickle.load(open(f"{data_dir}/kvld_matches_2.p", "rb"))
        except (OSError, IOError) as e:
            self.saved_matches = {}
            
        try:
            self.compute = pickle.load(open(f"{data_dir}/{self.solver}.p", "rb"))
        except (OSError, IOError) as e:
            self.compute = {}

        for _, _, localized_coord in self.compute.values():
            self.gmap3.scatter([localized_coord.latitude], [localized_coord.longitude], '#0000FF', size=7, marker=True)


        self.gmap3.draw(f"{data_dir}/image_locations_{self.solver}.html")
        self.run_metrics()
        
            
    def run_metrics(self):
        from metrics import process_data
        data_points = {k:v for k,v in self.compute.items() if k < 8100} # 6900
        print(f'{self.solver}: {process_data(data_points, self.solver)}, len: {len(self.compute)}')

    def iterate_frames(self):
        start_row = self.log.index[(self.log['frame_number'] == start_frame + 2490) & (self.log['new_frame'] == 1)].tolist()[0]
        for _, row in self.log.iloc[start_row:].iterrows():
            if row['new_frame'] == 1:
                start_time = time.time()
                _, frame = self.video.read()
                row['camera_matrix'] = self.format_camera_matrix(row)
                self.localize_frame(frame, row)
                # print(f'{self.solver}, Frame {self.frame_idx} took: {time.time() - start_time}')
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

    def match_frame_to_panorama(self, frame, metadata, n=4):
        fov = np.rad2deg(np.arctan(FRAME_WIDTH/metadata['focal_length_x']))

        if self.frame_idx in self.saved_matches and self.frame_idx not in self.compute:
            locations = []
            num_matches = []
            saved_match = self.saved_matches[self.frame_idx]
            kvld_matches = saved_match[0]
            metadata = saved_match[1]
            matches = []
           
            for (pano, camera_matrix), points1, points2, m in kvld_matches:
                K = camera_matrix
                matches.append([points1, points2])
                num_matches.append(len(points1))
                locations.append([pano.lat, pano.long])

            i = np.argpartition(num_matches, -n)[-n:]
            matches = np.array(matches, dtype=object)[i]
            locations = np.array(locations)[i]
            print(f'{self.solver} running frame {self.frame_idx}')
            if self.solver == 'scipy':
                frame_points, pano_points = find_correspondence_set_intersection(matches)
                estimate, error, localized_coord, locations = estimate_pose_with_3d_points(frame_points, pano_points, locations, metadata['course'], 12, 2.5, K)
            elif self.solver == 'g2o':
                estimate, error, localized_coord, locations = estimate_pose_with_3d_points_g2o(matches, locations, metadata['course'], 12, 2.5, K, metadata, fov, scaled_frame_width, scaled_frame_height)
            elif self.solver == 'ceres':
                estimate, error, localized_coord, locations = estimate_pose_with_3d_points_ceres(matches, locations, metadata['course'], 12, 2.5, K, metadata, fov, scaled_frame_width, scaled_frame_height)
            
            if localized_coord is not None:
                self.compute[self.frame_idx] = (estimate, error, localized_coord)
                self.frames_processed += 1

                if self.frames_processed % 10 == 0:
                    pickle.dump(self.compute, open(f"{data_dir}/{self.solver}.p", "wb"))
                    self.gmap3.scatter(locations[:, 0], locations[:, 1], '#FF0000', size=5, marker=True)
                    self.gmap3.scatter([localized_coord.latitude], [localized_coord.longitude], '#0000FF', size=7, marker=True)
                    self.gmap3.draw(f"{data_dir}/image_locations_{self.solver}.html")
                    self.run_metrics()
            
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
        return query(loc, n_points=10)
