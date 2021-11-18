from .util import get_existing_panoramas
from scipy import spatial
from config import data_dir

existing_panoramas = get_existing_panoramas()
pano_locs = [(p.lat, p.long) for p in existing_panoramas]
loc_to_pano = {(p.lat, p.long): p for p in existing_panoramas}
tree = spatial.KDTree(pano_locs)


def query(query_point, n_points=5, distance_upper_bound=0.001):
    result = tree.query([query_point], k=n_points, distance_upper_bound=distance_upper_bound)
    return [loc_to_pano[pano_locs[i]] for i in result[1][0] if i < len(pano_locs)]