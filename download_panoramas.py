import os
import requests
import requests_cache
import cv2
import urllib.error
from multiprocessing.pool import ThreadPool
import concurrent.futures

from download.streetview import fetch_panorama, fetch_metadata
from config import images_dir, sqlite_path

from download.util import separate_loc_list, combine_lat_long_lists, get_existing_panoramas, save_existing_panoramas
from download.waypoints import westwood_blvd, wilshire_blvd
from download.gpx_interpolate import gpx_interpolate

# Creates street_view_cache.sqlite if it doesn't already exist, reduces API usage
requests_cache.install_cache(sqlite_path, cache_control=False, expire_after=-1)

meta_base = 'https://maps.googleapis.com/maps/api/streetview/metadata?'
pic_base = 'https://maps.googleapis.com/maps/api/streetview?'
api_key = 'AIzaSyBjB2MBFlKlTIAbnG8D_t1oPqfObdR0xAA'  # TODO: Deactive the API key before we make the repo public

traj = wilshire_blvd
gpx_data = {'lat': separate_loc_list(traj)[0],
            'lon': separate_loc_list(traj)[1],
            'ele': [0 for x in traj],
            'tstamp': [0 for x in traj],
            'tzinfo': None}

# If num = 0, res determines spacing of points, deg must be [1, 5]
interpolated_points = combine_lat_long_lists(*gpx_interpolate(gpx_data, num=0, res=10, deg=1))

def request_metadata(lat, long):
    meta_params = {'key': api_key, 'location': f'{lat},{long}', 'source': 'outdoor'}
    meta_response = requests.get(meta_base, params=meta_params)
    if meta_response.ok:
        pano = fetch_metadata(meta_response.json()['pano_id'])
        print(f'Fetching meta for: {pano.pano_id}')
        return pano
    else:
        return None

def request_panorama(pano, idx, zoom):
    rgb = fetch_panorama(pano.pano_id, zoom)
    print(f'Fetched pano {pano.pano_id}, {idx}/{len(panos_to_get)}')
    fp = f'{images_dir}/{pano.pano_id}.jpg'
    cv2.imwrite(fp, cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    return pano

def get_meta():
    panoramas = set()
    print(f'Fetching metadata for {len(interpolated_points)} points')
    for (lat, long) in interpolated_points:
        pano = request_metadata(lat, long)
        panoramas.add(pano)

    return panoramas

existing_panos = get_existing_panoramas()
potential_panos = get_meta()
panos_to_get = potential_panos - existing_panos
save_every = 10
zoom = 5


for idx, pano in enumerate(panos_to_get):
    try:
        pano = request_panorama(pano, idx, zoom)
        existing_panos.add(pano)
        if idx % save_every == 0:
            save_existing_panoramas(existing_panos)
            print("Saved meta file!")
    except urllib.error.HTTPError as e:
        print(f'Error getting panorama for pano id: {pano.pano_id}\n{e}')

# for pano in panos_to_get:
#     if os.path.exists(f'{images_dir}/{pano.pano_id}.jpg'):
#         existing_panos.add(pano)

save_existing_panoramas(existing_panos)