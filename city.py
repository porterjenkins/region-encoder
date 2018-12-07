import numpy as np
from grid import GridCell

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
                    id_cnt += 1
        else:
            # TODO: implement other region types: e.g., tracts, etc...
            raise Exception("Compatibility with other region types not implemented yet.")


if __name__ == '__main__':

    chicago = City(name='chicago', lon_min=41.65021997246178, lon_max=42.02126162051242,
                   lat_min=-87.90448852338, lat_max=-87.53049651540705, sub_region_type='grid')

    chicago.init_sub_regions(n_splits=5)
