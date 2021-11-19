import os
import time
import pandas as pd

from localization.feature_matching import extract_features, match_frame_features_to_panoramas
from localization.segmentation import SemanticSegmentation
from download.query import query
from config import images_dir, start_frame, headings_, recording_dir
from itertools import islice
import cv2
from localization.visual_odometery import vo

FRAME_WIDTH = 640

class Vehicle:
    def __init__(self):
        self.log = pd.read_pickle(os.path.join(recording_dir, 'log.dat'))
        self.video = cv2.VideoCapture(os.path.join(recording_dir, 'Frames.m4v'))
        self.video.set(cv2.CAP_PROP_POS_FRAMES, start_frame-1)
        self.segmentation = SemanticSegmentation()
        self.vo = vo()
        self.traj = np.zeros(shape=(600, 800, 3)) #visualize trajectory



    def iterate_frames(self):
        start_row = self.log.index[(self.log['frame_number'] == start_frame + 2490) & (self.log['new_frame'] == 1)].tolist()[0]
        frame_idx = 0
        for _, row in self.log.iloc[start_row:].iterrows():
            if row['new_frame'] == 1:
                start_time = time.time()
                _, frame = self.video.read()
                self.localize_frame(frame, row)
                print(f'Frame {frame_idx} took: {time.time() - start_time}')
                frame_idx += 1

    def localize_frame(self, frame, metadata):
        # self.match_frame_to_panorama(frame, metadata)

        self.vo.process_frame(frame, metadata)
        coord = vo.t.flattern()
        print("x: {}, y: {}, z: {}".format(*[str(pt) for pt in coord]))
        draw_x, draw_y, draw_z = [int(round(x)) for x in coord]
        traj = cv.circle(traj, (draw_x + 400, draw_z + 100), 1, list((0, 255, 0)), 4)
        cv.imshow('trajectory', traj)
        #pull master and merge master

    def match_frame_to_panorama(self, frame, metadata):
        panoramas = self.get_nearby_panoramas(metadata)
        pano_data = self.extract_rectilinear_views(panoramas, metadata['course'])
        frame_data = self.process_frame(frame)
        matches = match_frame_features_to_panoramas(pano_data, frame_data)

        # for p, (l, h) in panoramas:
        #     cv2.imwrite(f'tmp/{p.pano_id}-{l}.jpg', cv2.imread(f'{images_dir}/{p.pano_id}-{l}.jpg'))
        #     cv2.imwrite(f'tmp/{p.pano_id}-{h}.jpg', cv2.imread(f'{images_dir}/{p.pano_id}-{h}.jpg'))

        for _, match in enumerate(islice(matches, 0, 50)):
            # print(f'Match with number of features: {match[-1]}')
            reference_img = cv2.cvtColor(match[1], cv2.COLOR_RGB2BGR)
            reference_img = cv2.drawMatchesKnn(frame_data[0], frame_data[1], reference_img, match[2], match[5], None, flags=2)
            cv2.imwrite(f'tmp/flann-match-{time.time_ns() - 1636597296826147000}.jpg', reference_img)

    def process_frame(self, frame):
        frame = cv2.resize(frame, (640, int(640*frame.shape[0]/frame.shape[1])), interpolation=cv2.INTER_AREA)
        frame = self.segmentation.segmentImage(frame)
        return frame, *extract_features(frame)

    def extract_rectilinear_views(self, panoramas, heading, pitch=10, fov=100, w=640, h=480):
        pano_data = []
        for pano in panoramas:
            pano_data.append([pano, pano.get_rectilinear_image(heading, pitch, fov, w, h)])
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
