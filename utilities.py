import cv2
import os
import pickle

from config import images_dir

def convert_keypoints_to_tuple(kp):
    index = []
    for point in kp:
        temp = (point.pt, point.size, point.angle, point.response, point.octave, 
            point.class_id) 
        index.append(temp)
    return index

def convert_tuple_to_keypoints(index):
    kp = []
    for point in index:
        temp = cv2.KeyPoint(x=point[0][0], y=point[0][1], size=point[1], angle=point[2], 
                                response=point[3], octave=point[4], class_id=point[5])
        kp.append(temp)
    return kp

def load_pano_features(pano_id):
    fp = os.path.join(images_dir, f'{pano_id}_features.dat')
    return pickle.load(open(os.path.join(images_dir, f'{pano_id}_features.dat'), 'rb'))