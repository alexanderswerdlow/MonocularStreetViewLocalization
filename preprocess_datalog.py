from stream.configuration import FRAMES_DEFAULT, GPS_DEFAULT, MOTION_DEFAULT
from stream.log import DataLog
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-data_dir', '--d', type=str, required=True)
parser.add_argument('-vid_freq', '--v', type=int, default=FRAMES_DEFAULT)
parser.add_argument('-imu_freq', '--i', type=int, default=MOTION_DEFAULT)
parser.add_argument('-gps_freq', '--g', type=int, default=GPS_DEFAULT)
args = parser.parse_args()


data_log = DataLog(args.d, args.v, args.i, args.g)

log = data_log.create_log()
log.to_pickle(os.path.join(args.d, 'log.dat'))

# import pandas as pd
# print(pd.read_pickle(os.path.join(args.d, 'log.dat')))