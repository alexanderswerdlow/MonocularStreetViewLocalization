from stream.configuration import FRAMES_DEFAULT, GPS_DEFAULT, MOTION_DEFAULT
from stream.log import DataLog
import os
from config import recording_dir

data_log = DataLog(recording_dir, FRAMES_DEFAULT, MOTION_DEFAULT, GPS_DEFAULT)
log = data_log.create_log()
log.to_pickle(os.path.join(recording_dir, 'log.dat'))

# import pandas as pd
# print(pd.read_pickle(os.path.join(args.d, 'log.dat')))