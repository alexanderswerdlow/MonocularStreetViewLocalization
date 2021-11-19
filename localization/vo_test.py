import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import os
from visual_odometry import visual_odometery




#Iphone params
# focal_length_x         1.426957e+03
# focal_length_y         1.426957e+03
# principal_point_x      9.547131e+02
# principal_point_y      7.281429e+02


image_dim = (1440, 1920, 3)
focal = 1426.957
pp = (954.7131, 728.1429)
lk_params = dict(winSize  = (21,21), criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))
detector=cv2.FastFeatureDetector_create(threshold=25, nonmaxSuppression=True)
min_features = 2000


debug = True
traj = np.zeros(shape=(600, 800, 3)) #visualize

vo = visual_odometery(focal, pp, lk_params, detector, min_features)

#TODO : Add flow lines
#TODO : check for mono coordinates
while(vo.hasNextFrame()):
	frame = vo.current_frame

	if debug:
		cv.imshow('frame', frame)
		k = cv.waitKey(1)
		if k == 27:
			break


	vo.process_frame()
	coord = vo.t.flattern()
	print("x: {}, y: {}, z: {}".format(*[str(pt) for pt in coord]))
	draw_x, draw_y, draw_z = [int(round(x)) for x in coord]
	traj = cv.circle(traj, (draw_x + 400, draw_z + 100), 1, list((0, 255, 0)), 4)
	cv.imshow('trajectory', traj)


# cv.imwrite("trajectory.png", traj)

cv.destroyAllWindows()


