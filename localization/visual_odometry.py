import numpy as np
import cv2
import matplotlib.pyplot as plt
 

#TODO : Convert images to PNG
#TODO : Add keypoint visualization
#TODO : Add flow lines visualization

#Iphone params
# focal_length_x         1.426957e+03
# focal_length_y         1.426957e+03
# principal_point_x      9.547131e+02
# principal_point_y      7.281429e+02
# image_dim              (1440, 1920, 3)


class vo:
	def __init__(self):
		self.lk_params = dict(winSize  = (21,21), criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))
		self.detector = cv2.FastFeatureDetector_create(threshold=25, nonmaxSuppression=True)
		self.R = np.zeros(shape=(3, 3))
		self.t = np.zeros(shape=(3, 3))
		self.n_features = 0
		self.min_features = 2000
		self.id = 0
		print("Testing VO")


	def detect(self, img):
		p0 = self.detector.detect(img)
		return np.array([x.pt for x in p0], dtype=np.float32).reshape(-1, 1, 2)


	def visual_odometery(self):
		
		if self.n_features < self.min_features: # Maintains at least min_features
			self.p0 = self.detect(self.old_frame)

		# Optical flow between frames, st holds status of points from frame to frame
		self.p1, st, err = cv2.calcOpticalFlowPyrLK(self.old_frame, self.current_frame, self.p0, None, **self.lk_params)
		
		self.good_old = self.p0[st == 1] # Save the good points from optical flow
		self.good_new = self.p1[st == 1]

		pp = (self.current_metadata['principal_point_x'], self.current_metadata['principal_point_y'])
		focal = self.current_metadata['focal_length_x']

		if self.id < 2:
			E, _ = cv2.findEssentialMat(self.good_new, self.good_old, focal, pp, cv2.RANSAC, 0.999, 1.0, None)
			_, self.R, self.t, _ = cv2.recoverPose(E, self.good_old, self.good_new, self.R, self.t, focal, pp, None)
		else:
			E, _ = cv2.findEssentialMat(self.good_new, self.good_old, focal, pp, cv2.RANSAC, 0.999, 1.0, None)
			_, R, t, _ = cv2.recoverPose(E, self.good_old, self.good_new, self.R.copy(), self.t.copy(), focal, pp, None)

			absolute_scale = self.get_absolute_scale()
			if (absolute_scale > 0.1 and abs(t[2][0]) > abs(t[0][0]) and abs(t[2][0]) > abs(t[1][0])):
				self.t = self.t + absolute_scale*self.R.dot(t)
				self.R = R.dot(self.R)
			else:
				print("not triggered")

		# Save the total number of good features
		self.n_features = self.good_new.shape[0]


	def get_absolute_scale(self):

		x_prev = self.old_metadata['ar_translation_x']
		y_prev = self.old_metadata['ar_translation_y']
		z_prev = self.old_metadata['ar_translation_z']
		x = self.current_metadata['ar_translation_x']
		y = self.current_metadata['ar_translation_y']
		z = self.current_metadata['ar_translation_z']
		curr_vect = np.array([[x], [y], [z]])
		prev_vect = np.array([[x_prev], [y_prev], [z_prev]])
		
		return np.linalg.norm(curr_vect - prev_vect)


	def process_frame(self, frame, metadata):

		if self.id == 0:
			self.current_frame = frame
			self.current_metadata = metadata
			self.id += 1
		else:
			self.old_frame = self.current_frame
			self.old_metadata = self.current_metadata
			self.current_frame = frame
			self.current_metadata = metadata
			self.visual_odometery()
			self.id += 1
