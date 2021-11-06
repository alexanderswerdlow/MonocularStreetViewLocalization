from download.util import Loc, separate_loc_list, get_existing_panoramas
from scipy import spatial
from config import data_dir

existing_panoramas = get_existing_panoramas()
pano_locs = [p.loc for p in existing_panoramas]
loc_to_pano = {p.loc: p for p in existing_panoramas}
tree = spatial.KDTree(pano_locs)


def query(query_point, n_points=5, distance_upper_bound=0.001):
    result = tree.query([query_point], k=n_points, distance_upper_bound=distance_upper_bound)
    return [loc_to_pano[Loc(*pano_locs[i])] for i in result[1][0] if i < len(pano_locs)]


def plot_query(query_point, closest_points):
    import gmplot
    gmap3 = gmplot.GoogleMapPlotter(34.061157672886466, -118.44550056779205, 17)
    gmap3.scatter(*separate_loc_list([p.loc for p in closest_points]), '#0000FF', size=5, marker=True)
    gmap3.scatter([query_point.lat], [query_point.long], '#FF0000', size=5, marker=True)
    gmap3.draw(f"{data_dir}/query_locations.html")


if __name__ == "__main__":
    query_point = Loc(34.06272153635474, -118.44537369759628)
    res = query(query_point)
    plot_query(query_point, res)
