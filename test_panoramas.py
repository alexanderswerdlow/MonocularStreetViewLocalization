from download.util import get_existing_panoramas
from config import images_dir
import cv2

panos = list(get_existing_panoramas())

for i in range(10):
    pano = panos[i]
    print(pano.pano_id)
    cv2.imshow('Panorama', cv2.imread(f'{images_dir}/{pano.pano_id}.jpg'))
    cv2.waitKey(0)
    rectilinear = pano.get_rectilinear_image(0, 10, 100)/255
    cv2.imshow('Rectilinear', rectilinear)
    cv2.waitKey(0)
    