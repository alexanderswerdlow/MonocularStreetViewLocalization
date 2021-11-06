import sys
import cv2
import signal

from vehicle import Vehicle
from localization.feature_matching import FeatureTracker

if len(sys.argv) == 1:
    print('Enter the path of the datalog to stream')
log = sys.argv[1]

test_vehicle = Vehicle(log)

signal.signal(signal.SIGINT, test_vehicle.close)

test_vehicle.start(10)