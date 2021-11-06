import multiprocessing
import threading

import os
import time
import cv2
import pandas as pd

from download.query import query
from download.util import Loc

class LocalizationProcess(multiprocessing.Process):
    def __init__(self, data_queue, location_queue):
        multiprocessing.Process.__init__(self)
        self.data_queue = data_queue
        self.location_queue = location_queue
    
    def run(self):
        print("Starting Localization Process")
        while True:
            frame, gps = self.data_queue.get() #blocking
            if frame is None:
                print("Exiting Localization Process")
                self.data_queue.task_done()
                break
            # localize frame
            self.localize(frame, gps)
            location = None # replace
            self.location_queue.put(location)
            self.data_queue.task_done()

    def localize(self, frame, gps):
        loc = Loc(gps[0], gps[1])
        panoramas = query(loc)
        print(panoramas)

class Vehicle:
    def __init__(self, data_dir):
        self.queried_panoramas = []
        self.data_queue = multiprocessing.JoinableQueue()
        self.location_queue = multiprocessing.Queue()

        self.log = pd.read_pickle(os.path.join(data_dir, 'log.dat'))
        self.video = cv2.VideoCapture(os.path.join(data_dir, 'Frames.m4v'))

        self.exit_thread = False

    def start(self, stream_freq):
        self.localization_process = LocalizationProcess(self.data_queue, self.location_queue)
        self.localization_process.start()
        self.stream_log(stream_freq)

        print("Waiting to Join Localization Process")
        self.data_queue.join()
        self.localization_process.join()
        print("Terminated Localization Process")

    def stream_log(self, frequency):
        for index, row in self.log.iterrows():
            if self.exit_thread:
                return

            gps = [row['latitude'], row['longitude']]
            # read other sensor data here
            if row['new_frame'] == 1:
                print('New Frame!')
                ret, frame = self.video.read()
                self.data_queue.put([frame, gps])
            time.sleep(1/frequency)

    def close(self, sig, frame):
        self.exit_thread = True
        self.localization_process.kill()