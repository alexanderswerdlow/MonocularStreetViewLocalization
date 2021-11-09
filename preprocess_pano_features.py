import os
import cv2
import numpy as np
import pickle

from config import images_dir
from utilities import convert_keypoints_to_tuple
from download.util import get_existing_panoramas
from localization.feature_matching import FeatureTracker

feature_tracker = FeatureTracker()
panoramas = get_existing_panoramas()
headings = [0, 45, 90, 135, 180, 225, 270, 315]

i = 1
for pano in panoramas:
    print(f'Extracting features for panorama ID: {pano.pano_id}, {i}/{len(panoramas)}')
    feature_dict = {}
    for heading in headings:
        fp = os.path.join(images_dir, f'{pano.pano_id}-{heading}.jpg')
        img = cv2.imread(fp)
        # img = cv2.resize(img, (1920, 1920), interpolation=cv2.INTER_LANCZOS4)
        feature_tracker.extract_features(img, show_keypoints=False)
        kp, des = feature_tracker.current_frame_features

        feature_dict[heading] = (convert_keypoints_to_tuple(kp), des)
    pickle.dump(feature_dict, open(os.path.join(images_dir, f'{pano.pano_id}_features.dat'), 'wb'))
    i += 1
