from util import Loc, separate_loc_list, get_existing_panoramas
from scipy import spatial

existing_panoramas = get_existing_panoramas()
pano_locations = [tuple(pano.loc) for pano in existing_panoramas]
tree = spatial.KDTree(pano_locations)

def query(query_point, n_points = 10, distance_upper_bound=0.001):
    result = tree.query([query_point], k = n_points, distance_upper_bound=distance_upper_bound)
    return [Loc(*pano_locations[i]) for i in result[1][0] if i < len(pano_locations)]

def plot_query(query_point, closest_points):
    import gmplot
    gmap3 = gmplot.GoogleMapPlotter(34.061157672886466, -118.44550056779205, 17)
    gmap3.scatter(*separate_loc_list(closest_points), '#0000FF', size=5, marker=True)
    gmap3.scatter([query_point.lat],[query_point.long] ,'#FF0000', size=5, marker=True)
    gmap3.draw("download/query_locations.html")

if __name__ == "__main__":
    query_point = Loc(34.06272153635474, -118.44537369759628)
    plot_query(query_point, query(query_point))