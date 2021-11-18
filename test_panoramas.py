from download.util import get_existing_panoramas
import cv2

panos = list(get_existing_panoramas())

for i in range(10):
    pano = panos[i]
    print(pano.pano_id)
    cv2.imshow('Panorama', cv2.imread(pano.image_fp))
    cv2.waitKey(0)
    rectilinear = pano.get_rectilinear_image(0, 10, 100)/255
    cv2.imshow('Rectilinear', cv2.cvtColor(rectilinear, cv2.COLOR_RGB2BGR))
    cv2.waitKey(0)
    