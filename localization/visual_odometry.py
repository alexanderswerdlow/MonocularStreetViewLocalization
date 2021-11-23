import numpy as np
import cv2
import matplotlib.pyplot as plt
 



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
			# print("abs scale", absolute_scale)
			if (absolute_scale > 0.10 and abs(t[2][0]) > abs(t[0][0]) and abs(t[2][0]) > abs(t[1][0])):
				self.t = self.t + absolute_scale*self.R.dot(t)
				self.R = R.dot(self.R)
			else:
				pass
				# print("not triggered", absolute_scale, t[2][0], t[1][0], t[0][0], (abs(t[2][0]) > abs(t[0][0])), (abs(t[2][0]) > abs(t[1][0])))

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





# new_frame              1.000000e+00
# frame_timestamp        1.636579e+09
# frame_number           7.504000e+03
# focal_length_x         1.427997e+03
# focal_length_y         1.427997e+03
# principal_point_x      9.548727e+02
# principal_point_y      7.281066e+02
# motion_timestamp       1.636579e+09

# rotation_rate_x       -1.755000e-03
# rotation_rate_y       -1.070100e-02
# rotation_rate_z       -7.440000e-04

# gravity_x             -9.328870e-01
# gravity_y             -5.571500e-02
# gravity_z              3.558350e-01

# user_accel_x           2.305100e-02
# user_accel_y           1.126600e-02
# user_accel_z          -1.382450e-01

# motion_heading         2.865248e+02

# gps_timestamp          1.636579e+09
# latitude               3.406030e+01
# longitude             -1.184380e+02
# horizontal_accuracy    1.423900e+01
# altitude               1.047715e+02
# vertical_accuracy      1.008324e+01
# floor_level            0.000000e+00
# course                 7.258802e+01
# speed                  1.337058e+01

# ar_timestamp           1.636579e+09
# ar_translation_x      -1.330683e+03
# ar_translation_y      -8.134804e+00
# ar_translation_z      -1.776376e+02
# ar_quaternion_w        3.710970e-01
# ar_quaternion_x        4.053900e-02
# ar_quaternion_y        9.117030e-01
# ar_quaternion_z       -1.715860e-01
