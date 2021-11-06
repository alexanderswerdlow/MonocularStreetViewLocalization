import csv
import os
from .configuration import *
import pandas as pd
import sys

VID = 'VID'
IMU = 'IMU'
GPS = 'GPS'
AR  = 'AR'

class SensorData:
    def __init__(self, data_dir, file, freq, default, stype):
        self.f = open(os.path.join(data_dir, file), 'r')
        self.reader = csv.reader(self.f, quoting=csv.QUOTE_NONNUMERIC)
        self.freq = freq
        self.type = stype

        self.default_freq = default
        self.ratio = 1 / self.freq

class DataLog:
    """
    Usage:
    Create DataLog object with:
        data_dir : path to directory containing Frames.txt, MotARH.txt, GPS.txt
        vid_freq : default 30
        imu_freq : default 100
        gps_freq : default 1
    Call create_log() to create pandas dataframe with compiled information
    Each row of dataframe has data from all 3 sensors, along with a flag 
        which is set on for every new video frame
    Frequency is based on the highest set freqency when creating datalog
    Other 2 sensors will have the reading at the closest following timestamp
    """
    def __init__(self, data_dir, vid_freq=FRAMES_DEFAULT, imu_freq=MOTION_DEFAULT, 
                 gps_freq=GPS_DEFAULT, ar_freq=AR_DEFAULT):

        self.sensors = [
            SensorData(data_dir, FRAMES_CSV, vid_freq, FRAMES_DEFAULT, VID),
            SensorData(data_dir, MOTION_CSV, imu_freq, MOTION_DEFAULT, IMU),
            SensorData(data_dir, GPS_CSV, gps_freq, GPS_DEFAULT, GPS),
            SensorData(data_dir, AR_CSV, ar_freq, AR_DEFAULT, AR)
        ]

        # sensor with highest reading frequency
        self.ref = max(self.sensors, key=lambda k: k.freq)

        # other sensors
        self.others = [s for s in self.sensors if s is not self.ref]

        self.vid_ref = self.ref.type == VID
        self.log = []

    def read_until(self, reader, cache, timestamp):
        """
        return first line after given timestamp from specified csv reader
        """
        line = cache
        while reader and timestamp > float(line[0]):
            try:
                line = next(reader)
            except StopIteration:
                break
        return line

    def create_log(self):
        """
        compile data from video, motion, gps into dataframe
        """
        ref_line = next(self.ref.reader)
        other_lines = [next(s.reader) for s in self.others]

        timestamp = float(ref_line[0])
        last_frame = -1
        dup_count = 0
        while self.ref.reader:
            # reference reading
            new_line = self.read_until(self.ref.reader, ref_line, timestamp)

            # condition to end looping
            if ref_line == new_line:
                dup_count += 1 
                if dup_count > self.ref.freq / self.ref.default_freq:
                    break
            else:
                dup_count = 0
            ref_line = new_line

            # get readings from other 2 sensors closest to current timestamp
            other_lines = [self.read_until(z[0], z[1], timestamp) 
                         for z in zip([s.reader for s in self.others], other_lines)]

            # increment timestamp based on highest frequency
            timestamp += self.ref.ratio

            vid_line = self.vid_ref and ref_line or other_lines[0]
            imu_line = self.vid_ref and other_lines[0] or ref_line
            gps_line = other_lines[1]
            ar_line  = other_lines[2]

            # set new frame flag 
            frame_flag = [0]
            if int(vid_line[1]) != last_frame:
                frame_flag = [1]
            last_frame = int(vid_line[1])

            self.log.append(frame_flag + vid_line + imu_line + gps_line + ar_line)
        
        # convert log to dataframe
        log_columns = ['new_frame'] + FRAMES_FIELDS + MOTION_FIELDS + GPS_FIELDS + AR_FIELDS
        return pd.DataFrame(self.log, columns=log_columns)
