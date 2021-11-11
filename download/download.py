import gmplot
import pyproj
import requests_cache
import requests
from download.util import Pano, Loc, separate_loc_list, combine_lat_long_lists, get_existing_panoramas, save_existing_panoramas
from download.gpx_interpolate import gpx_interpolate
from download import streetview
from PIL import Image
from download.depth import get_depth_map
from download.waypoints import westwood_blvd, whilshire_blvd
from config import images_dir, sqlite_path, data_dir

meta_base = 'https://maps.googleapis.com/maps/api/streetview/metadata?'
pic_base = 'https://maps.googleapis.com/maps/api/streetview?'
api_key = 'AIzaSyBjB2MBFlKlTIAbnG8D_t1oPqfObdR0xAA'  # TODO: Deactive the API key before we make the repo public

geodesic = pyproj.Geod(ellps='WGS84')
fov, img_w, img_h = 90, 640, 640

# Creates street_view_cache.sqlite if it doesn't already exist, reduces API usage
requests_cache.install_cache(sqlite_path, cache_control=False, expire_after=-1)

traj = whilshire_blvd
gpx_data = {'lat': separate_loc_list(traj)[0],
            'lon': separate_loc_list(traj)[1],
            'ele': [0 for x in traj],
            'tstamp': [0 for x in traj],
            'tzinfo': None}

# If num = 0, res determines spacing of points, deg must be [1, 5]
interpolated_points = combine_lat_long_lists(*gpx_interpolate(gpx_data, num=0, res=10, deg=1))


def get_meta():
    panoramas = set()
    for idx, (lat, long) in enumerate(interpolated_points):
        meta_params = {'key': api_key, 'location': f'{lat},{long}', 'source': 'outdoor'}
        meta_response = requests.get(meta_base, params=meta_params)
        if meta_response.ok:
            if idx == len(interpolated_points) - 1:
                fwd_azimuth, _, _ = geodesic.inv(interpolated_points[idx - 1][1], interpolated_points[idx - 1][0], long, lat)
            else:
                fwd_azimuth, _, _ = geodesic.inv(long, lat, interpolated_points[idx + 1][1], interpolated_points[idx + 1][0])

            if fwd_azimuth < 0:
                fwd_azimuth += 360

            pano = Pano(Loc(meta_response.json()['location']['lat'], meta_response.json()['location']['lng']), meta_response.json()['pano_id'])
            panoramas.add(pano)
            if not meta_response.from_cache:
                print(f"Meta not from cache: f'{lat},{long}'")
    return panoramas


def get_depth_maps(p):
    # im = view_depth_map(get_depth_map(pano_id=pano.pano_id))
    # plt.imsave(f'{images_dir}/{pano.get_name()}-depth.png', im)
    # existing_panos.add(pano)

    # im = depthinfo_to_image()
    # im.save(f'{images_dir}/{pano.get_name()}-depth.png')
    # xform.cut_tiles_and_package_to_zip(im, "dpth", '1I3aa1QeAHLC7b5Ar12nWg', "png")
    existing_panos.add(p._replace(depth_map=get_depth_map(pano_id=p.pano_id)))


def get_unofficial(p):
    panorama = streetview.download_panorama_v3(p.pano_id)
    im = Image.fromarray(panorama)
    im.save(f'{images_dir}/{p.get_name()}.png')


def get_official(p):
    _, pano_id, _ = p
    for i in range(8):
        pic_params = {'key': api_key, 'pano': pano_id, 'size': f'{img_w}x{img_h}', 'fov': fov, 'heading': str(i * 45), 'pitch':'30'}
        with requests.get(pic_base, params=pic_params) as pic_response:
            if not pic_response.from_cache:
                print(f"Image not from cache: {pano_id}")

            if pic_response.ok:
                with open(f'{images_dir}/{p.get_name()}-{pic_params["heading"]}.jpg', 'wb') as file:
                    file.write(pic_response.content)

def plot(panos):
    gmap3 = gmplot.GoogleMapPlotter(34.061157672886466, -118.44550056779205, 17, apikey=api_key)
    gmap3.scatter(*separate_loc_list(list(set([x.loc for x in panos]))), '#FF0000', size=5, marker=True)
    # gmap3.plot(*separate_loc_list([x.loc for x in existing_panoramas]), 'cornflowerblue', edge_width=2.5)
    gmap3.draw(f"{data_dir}/image_locations.html")

existing_panos = get_existing_panoramas()
potential_panos = get_meta()
panos_to_get = potential_panos - existing_panos  # Don't redownload images already present

print(f'Potential: {len(potential_panos)}, Existing: {len(existing_panos)}, To Get: {len(panos_to_get)}')

for idx, pano in enumerate(sorted(panos_to_get)):
    get_official(pano)
    # get_official(pano)
    get_depth_maps(pano)
    existing_panos.add(pano)
    if idx % 10 == 0:
        save_existing_panoramas(existing_panos)
        print("Saved meta file!")

save_existing_panoramas(existing_panos)
plot(existing_panos)
