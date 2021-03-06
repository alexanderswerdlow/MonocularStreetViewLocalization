import numpy as np
from numpy.lib.npyio import save
import requests
import imageio
import xmltodict
from config import images_dir, use_pickled_images
import pickle
import cv2
import os

from .depth import decompress_raw_depth_map
from .backprojection import backprojection_rectification

tile_width = 512
tile_height = 512

pano_url = 'http://maps.google.com/cbk?output=tile&panoid={0}&zoom={1}&x={2}&y={3}'
meta_url = 'http://cbk0.google.com/cbk?output=xml&panoid={0}&dm=1'

if use_pickled_images:
    images = pickle.load(open(f"{images_dir}/images.p", "rb"))
     
def save_pickled_images():
    images = {j:cv2.imread(f'{images_dir}/{j}.jpg') for j in [f[:-4] for f in os.listdir(images_dir) if f.endswith('.jpg')]}
    pickle.dump(images, open(f"{images_dir}/images.p", "wb"))
    exit()

def get_image(pano_id):
    return images[pano_id] if use_pickled_images else cv2.imread(f'{images_dir}/{pano_id}.jpg')

class Pano:
    def __init__(self, lat, long, pano_id, depth_map, projection):
        self.lat = lat
        self.long = long
        self.pano_id = pano_id
        self.depth_map = depth_map
        self.projection = projection

    def get_rectilinear_image(self, heading, pitch, fov, w=640, h=480):
        yaw = float(self.projection['@pano_yaw_deg'])
        rectilinear = backprojection_rectification(get_image(self.pano_id), yaw, fov, heading, pitch, w, h)
        return rectilinear.astype(np.uint8)

    def get_rectilinear_depth(self, heading, pitch, fov, w=640, h=480):
        yaw = float(self.projection['@pano_yaw_deg'])
        depth_map = np.flip(cv2.resize(self.depth_map, (1920, 1440), interpolation=cv2.INTER_NEAREST), axis=1)
        rectilinear = backprojection_rectification(depth_map, yaw, fov, heading, pitch, w, h)
        return rectilinear

    def get_depth_point_cloud(self, heading, pitch, fov, K, w=640, h=480):
        rectilinear = self.get_rectilinear_depth(self, heading, pitch, fov, w, h)
        cx = K[0, 2]
        cy = K[1, 2]
        fx = K[0, 0]
        fy = K[1, 1]

        points = []
        for u in range(w):
            for v in range(h):
                z = rectilinear[v, u]
                if z != np.nan:
                    x = z * ((u - cx) / fx)
                    y = z * ((v - cy) / fy)
                    points.append([x, y, z])

        return np.array(points), rectilinear

    def __hash__(self):
        return hash(self.pano_id)

    def __eq__(self, other):
        return isinstance(other, Pano) and self.pano_id == other.pano_id

    def get_name(self):
        return self.pano_id


def fetch_panorama(pano_id, zoom):
    cols = np.power(2, zoom)
    rows = np.power(2, zoom-1)
    panorama = np.zeros((tile_height * rows, tile_width * cols, 3), dtype=np.float32)

    for y in range(rows):
        for x in range(cols):
            tile = imageio.imread(pano_url.format(pano_id, zoom, x, y))
            panorama[y*tile_height:(y+1)*tile_height, x*tile_width:(x+1)*tile_width] = tile

    return panorama


def fetch_metadata(pano_id):
    response = requests.get(meta_url.format(pano_id))
    metadata = xmltodict.parse(response.content)['panorama']
    depth_map = decompress_raw_depth_map(metadata['model']['depth_map'])
    projection = dict(metadata['projection_properties'])
    lat = metadata['data_properties']['@lat']
    long = metadata['data_properties']['@lng']

    return Pano(float(lat), float(long), pano_id, depth_map, projection)
