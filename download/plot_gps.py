from download.util import Loc, separate_loc_list, get_existing_panoramas
from scipy import spatial
from config import data_dir
import gmplot

class CustomGoogleMapPlotter(gmplot.GoogleMapPlotter):
    def __init__(self, center_lat, center_lng, zoom, apikey='',
                 map_type='hybrid'):
        super().__init__(center_lat, center_lng, zoom, apikey)

        self.map_type = map_type
        assert(self.map_type in ['roadmap', 'satellite', 'hybrid', 'terrain'])

    def write_map(self,  f):
        f.write('\t\tvar centerlatlng = new google.maps.LatLng(%f, %f);\n' %
                (self.center[0], self.center[1]))
        f.write('\t\tvar myOptions = {\n')
        f.write('\t\t\tzoom: %d,\n' % (self.zoom))
        f.write('\t\t\tcenter: centerlatlng,\n')

        # This is the only line we change
        f.write('\t\t\tmapTypeId: \'{}\'\n'.format(self.map_type))


        f.write('\t\t};\n')
        f.write(
            '\t\tvar map = new google.maps.Map(document.getElementById("map_canvas"), myOptions);\n')
        f.write('\n')


gps = '/Volumes/GoogleDrive/Shared drives/EE209AS/data/recordings/2021-11-10T13-16-47/GPS.txt'
from csv import reader
lats, longs = [], []
# open file in read mode
with open(gps, 'r') as read_obj:
    # pass the file object to reader() to get the reader object
    csv_reader = reader(read_obj)
    # Iterate over each row in the csv using reader object
    for row in csv_reader:
        # row variable is a list that represents a row in csv
        lats.append(float(row[1]))
        longs.append(float(row[2]))


gmap3 = CustomGoogleMapPlotter(34.061157672886466, -118.44550056779205, 17)
gmap3.scatter(lats, longs, '#0000FF', size=5, marker=True)
gmap3.draw(f"gps_locations.html")


