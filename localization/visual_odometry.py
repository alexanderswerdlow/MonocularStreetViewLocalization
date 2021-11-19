import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt


#TODO : Add stream data (frames and AR poses)
#TODO : Convert images to PNG


class visual_odometery:
	def __init__(self, 
				focal_length, # focal length of camera 
				pp, # principal point of camera
				lk_params # Lucus Kanade Optical flow parameters
				detector
				min_features): #Feature detector

		self.focal = focal_length
		self.pp = pp
		self.lk_params = lk_params
		self.detector = detector
        self.R = np.zeros(shape=(3, 3))
        self.t = np.zeros(shape=(3, 3))
        self.n_features = 0
        self.min_features = min_features
        self.id = 0

    def hasNextFrame(self):
    	#TODO : check data streaming
        return self.id < len(os.listdir(self.file_path)) 


    def detect(self, img):
    	#TODO : Debug to add keypoint visualization
        p0 = self.detector.detect(img)
        return np.array([x.pt for x in p0], dtype=np.float32).reshape(-1, 1, 2)


    def visual_odometery(self):
    	#TODO : Check id and absolute_scale
        # Maintains at least min_features
        if self.n_features < self.min_features:
            self.p0 = self.detect(self.old_frame)

        # Optical flow between frames, st holds status of points from frame to frame
        self.p1, st, err = cv2.calcOpticalFlowPyrLK(self.old_frame, self.current_frame, self.p0, None, **self.lk_params)
        
        # Save the good points from the optical flow
        self.good_old = self.p0[st == 1]
        self.good_new = self.p1[st == 1]


        # If the frame is one of first two, we need to initalize
        # our t and R vectors so behavior is different
        if self.id < 2:
            E, _ = cv2.findEssentialMat(self.good_new, self.good_old, self.focal, self.pp, cv2.RANSAC, 0.999, 1.0, None)
            _, self.R, self.t, _ = cv2.recoverPose(E, self.good_old, self.good_new, self.R, self.t, self.focal, self.pp, None)
        else:
            E, _ = cv2.findEssentialMat(self.good_new, self.good_old, self.focal, self.pp, cv2.RANSAC, 0.999, 1.0, None)
            _, R, t, _ = cv2.recoverPose(E, self.good_old, self.good_new, self.R.copy(), self.t.copy(), self.focal, self.pp, None)

            absolute_scale = self.get_absolute_scale()
            if (absolute_scale > 0.1 and abs(t[2][0]) > abs(t[0][0]) and abs(t[2][0]) > abs(t[1][0])):
                self.t = self.t + absolute_scale*self.R.dot(t)
                self.R = R.dot(self.R)

        # Save the total number of good features
        self.n_features = self.good_new.shape[0]


    def get_absolute_scale(self):
    	# TODO : Get current AR pose data
        # pose = self.pose[self.id - 1].strip().split() store previous AR pose
        # x_prev = float(pose[3])
        # y_prev = float(pose[7])
        # z_prev = float(pose[11])
        # pose = self.pose[self.id].strip().split() get new AR pose
        # x = float(pose[3])
        # y = float(pose[7])
        # z = float(pose[11])

        curr_vect = np.array([[x], [y], [z]])
        prev_vect = np.array([[x_prev], [y_prev], [z_prev]])
        
        return np.linalg.norm(curr_vect - prev_vect)


    def process_frame(self):
    	#TODO : Get current video frame
        if self.id < 2:
            # self.old_frame = cv2.imread(self.file_path +str().zfill(6)+'.png', 0)
            # self.current_frame = cv2.imread(self.file_path + str(1).zfill(6)+'.png', 0)
            self.visual_odometery()
            self.id = 2
        else:
            self.old_frame = self.current_frame
            # self.current_frame = cv2.imread(self.file_path + str(self.id).zfill(6)+'.png', 0)
            self.visual_odometery()
            self.id += 1
