import os
import requests
import requests_cache
import cv2
import urllib.error
from multiprocessing.pool import ThreadPool
import concurrent.futures

from download.streetview import fetch_panorama, fetch_metadata
from config import images_dir, sqlite_path, api_key

from download.util import separate_loc_list, combine_lat_long_lists, get_existing_panoramas, save_existing_panoramas
from download.waypoints import new_whilshire, wilshire_blvd, westwood_blvd
from download.gpx_interpolate import gpx_interpolate

# Creates street_view_cache.sqlite if it doesn't already exist, reduces API usage
requests_cache.install_cache(sqlite_path, cache_control=False, expire_after=-1)

meta_base = 'https://maps.googleapis.com/maps/api/streetview/metadata?'
pic_base = 'https://maps.googleapis.com/maps/api/streetview?'

traj = westwood_blvd
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
save_every = 1
zoom = 5

def down(pano):
    try:
        if not os.path.exists(f'{images_dir}/{pano.pano_id}.jpg'):
            request_panorama(pano, 1, zoom)
            print('done')
        # # existing_panos.add(pano)
        # if idx % save_every == 0:
        #     save_existing_panoramas(existing_panos)
        #     print("Saved meta file!")
    except urllib.error.HTTPError as e:
        print(f'Error getting panorama for pano id: {pano.pano_id}\n{e}')

from multiprocessing import Pool
print(f"getting: {len(panos_to_get)}")

for p in panos_to_get:
    existing_panos.add(p)

save_existing_panoramas(existing_panos)
print("starting")

with Pool(10) as p:
    print(p.map(down, panos_to_get))


# for pano in potential_panos:
#     if os.path.exists(f'{images_dir}/{pano.pano_id}.jpg'):
#         existing_panos.add(pano)

save_existing_panoramas(existing_panos)