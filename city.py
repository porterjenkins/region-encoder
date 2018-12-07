import numpy as np
from grid import GridCell
from geopy.distance import distance
from config import get_config

class City(object):

    def __init__(self, lon_min, lon_max, lat_min, lat_max, name, sub_region_type='grid'):

        self.lat_min = lat_min
        self.lat_max = lat_max
        self.lon_min = lon_min
        self.lon_max = lon_max
        self.name = name
        self.sub_region_type = sub_region_type

        self.sub_regions = dict()

        self.lat_cut_points = np.array
        self.lon_cut_points = np.array


    def init_sub_regions(self, n_splits=None):

        if self.sub_region_type == 'grid' and n_splits is not None:
            lat_cut_points = np.linspace(start=self.lat_min, stop=self.lat_max, num=n_splits+1)
            lon_cut_points = np.linspace(start=self.lon_min, stop=self.lon_max, num=n_splits+1)

            id_cnt = 0
            for i in range(n_splits):
                for j in range(n_splits):
                    cell = GridCell(id=id_cnt,
                                    lat_range=(lat_cut_points[i], lat_cut_points[i + 1]),
                                    lon_range=(lon_cut_points[j], lon_cut_points[j + 1]))

                    self.sub_regions[id_cnt] = cell
                    cell.compute_distance()
                    id_cnt += 1

        else:
            # TODO: implement other region types: e.g., tracts, etc...
            raise Exception("Compatibility with other region types not implemented yet.")


    def compute_distance(self):
        lon_dist = distance(self.lon_min, self.lon_max)
        lat_dist = distance(self.lat_min, self.lat_max)

        print("Longitude Dist: {}".format(lon_dist))
        print("Latitude Dist: {}".format(lat_dist))




if __name__ == '__main__':

    config = get_config()

    chicago = City(name='chicago',
                   lon_min=config['lon_min'],
                   lon_max=config['lon_max'],
                   lat_min=config['lat_min'],
                   lat_max=config['lat_max'],
                   sub_region_type='grid')

    #chicago.init_sub_regions(n_splits=20)
