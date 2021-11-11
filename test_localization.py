import sys
import cv2
import signal

from vehicle import Vehicle
from localization.feature_matching import FeatureTracker
from config import recording_dir

if __name__ == '__main__':
    test_vehicle = Vehicle(recording_dir)
    signal.signal(signal.SIGINT, test_vehicle.close)
    test_vehicle.start(100)