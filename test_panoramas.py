from download.util import get_existing_panoramas
from config import images_dir, data_dir
import numpy as np
import cv2

panos = np.array(list(get_existing_panoramas()))
pano_id = '-p_u9lSmJYJjipcCNx3LQA'

pano = panos[[p.pano_id == pano_id for p in panos]][0]

cv2.imshow('Panorama', cv2.imread(f'{images_dir}/{pano.pano_id}.jpg'))
cv2.waitKey(0)
rectilinear = pano.get_rectilinear_image(90, 10, 100)/255
cv2.imshow('Rectilinear', rectilinear)
cv2.waitKey(0)

cv2.imwrite(f'{data_dir}/test.png', rectilinear*255)
