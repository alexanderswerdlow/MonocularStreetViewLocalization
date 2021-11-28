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
from localization.localization import estimate_location, find_homography, find_correspondence_set_intersection, estimate_pose_with_3d_points, estimate_pose_with_3d_points_g2o
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
        self.measurements = []

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
                # print(f'Frame {self.frame_idx} took: {time.time() - start_time}')
                self.frame_idx += 1

    def format_camera_matrix(self, metadata):
        camera_matrix = np.array([
            [metadata['focal_length_x'], 0, metadata['principal_point_x']],
            [0, metadata['focal_length_y'], metadata['principal_point_y']],
            [0, 0, 0]
        ])/SCALE_FACTOR
        return camera_matrix

    def localize_frame(self, frame, metadata):

        # self.measurements = pickle.load(open(f"save.p", "rb"))

        # from config import api_key
        # from pykalman import KalmanFilter
        # from localization.localization import CustomGoogleMapPlotter
        # kf = KalmanFilter(initial_state_mean=self.measurements[0], n_dim_obs=2)
        # measurements_ =   np.asarray(self.measurements)
        # meas = kf.em(measurements_, n_iter=50).smooth(measurements_)[0]
        # gmap3_g2o = CustomGoogleMapPlotter(34.060458, -118.437621, 17, apikey=api_key)
        # gmap3_g2o.scatter(meas[:, 0], meas[:, 1], '#0000FF', size=7, marker=True)
        # gmap3_g2o.draw(f"tmp/image_locations.html")
        # exit()
                
        self.match_frame_to_panorama(frame, metadata)
        # Do Visual Odometry

    def match_frame_to_panorama(self, frame, metadata, n=4):
        fov = np.rad2deg(np.arctan(FRAME_WIDTH/metadata['focal_length_x']))

        # if self.frame_idx % 10 == 0 and self.frame_idx not in self.saved_matches:
        #     print(f'Starting on Frame: {self.frame_idx}')
        #     panoramas = self.get_nearby_panoramas(metadata)
        #     pano_data = self.extract_rectilinear_views(panoramas, metadata, fov)
        #     frame_data = self.process_frame(frame)

        #     pano_dict = {p[0].pano_id: p for p in pano_data}
        #     kvld_matches = get_kvld_matches((self.frame_idx, frame_data), pano_dict)
        # print(kvld_matches)
        #     self.saved_matches[self.frame_idx] = (kvld_matches, metadata)
        #     pickle.dump(self.saved_matches, open("westwood_matches.p", "wb"))

        # return None

        if self.frame_idx in self.saved_matches:
            print(f"Starting Frame {self.frame_idx}")
            locations = []
            num_matches = []
            saved_match = self.saved_matches[self.frame_idx]
            kvld_matches = saved_match[0]
            metadata = saved_match[1]
            matches = []
            K = np.zeros((3, 3))
            for (pano, camera_matrix), points1, points2, m in kvld_matches:
                K = camera_matrix
                # points1 = [(int(x), int(y)) for x, y in points1]
                # points2 = [(int(x), int(y)) for x, y in points2]
                matches.append([points1, points2])
                
                num_matches.append(len(points1))
                locations.append([pano.lat, pano.long])
                # breakpoint()
                # R, t, points = find_homography(points1, points2, camera_matrix, cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY), cv2.cvtColor(pano.get_rectilinear_image(metadata['course'], 12, fov), cv2.COLOR_BGR2GRAY))
                # breakpoint()
                # translations.append([t[2], t[0]])

            i = np.argpartition(num_matches, -n)[-n:]
            matches = np.array(matches)[i]
            locations = np.array(locations)[i]
            # breakpoint()
            
            
            # directions = self.get_angles(np.array(translations)[i], metadata['course'])
            # localized_point = estimate_location(locations, directions)
            # print(localized_point)
            frame_points, pano_points = find_correspondence_set_intersection(matches)
            if len(frame_points) < 6:
                return None
            pts3D = find_homography(pano_points[0], pano_points[1], K, None, None)
            # estimate_pose_with_3d_points(frame_points, pano_points, locations, metadata['course'], 12, 2.5, K)
            ret = estimate_pose_with_3d_points_g2o(frame_points, pano_points, locations, metadata['course'], 12, 2.5, K, metadata, pts3D)
            if ret is not None:
                self.measurements.append([ret[0], ret[1]])

        #if self.frame_idx % 100 == 0 and len(self.measurements) > 2:
        #     # from config import api_key
        #     # from pykalman import KalmanFilter
        #     # from localization.localization import CustomGoogleMapPlotter
        #     # kf = KalmanFilter(initial_state_mean=self.measurements[0], n_dim_obs=2)
        #     # measurements_ =   np.asarray(self.measurements)
        #     # meas = kf.em(measurements_, n_iter=2).smooth(measurements_)[0]
        #     # gmap3_g2o = CustomGoogleMapPlotter(34.060458, -118.437621, 17, apikey=api_key)
        #     # gmap3_g2o.scatter(meas[:, 0], meas[:, 1], '#0000FF', size=7, marker=True)
        #     # gmap3_g2o.draw(f"tmp/image_locations.html")
        #    pickle.dump(self.measurements, open("save.p", "wb"))
                    
            # 1. Find 3d coordinates from just the panoramas. Initial guess is just triangulation from 2 panos
            #    We know 6DOF pose for each pano and image points (all image points for each pano is sorted relative to its
            #    corresponding frame point), so we can calculate the 3d points (apply a solver)
            # 2. PnP solver to find frame points pose w.r.t 3d points

            
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
