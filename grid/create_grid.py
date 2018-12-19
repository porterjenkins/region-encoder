import pickle

import numpy

from config import get_config


class RegionGrid:

    def __init__(self, poi_file, grid_size=100):

        poi = pickle.load(poi_file)
        rect, self.categories = RegionGrid.handle_poi(poi)
        self.lon_min, self.lon_max, self.lat_min, self.lat_max = rect["lon_min"], rect["lon_max"], rect["lat_min"], \
                                                                 rect["lat_max"]
        self.grid_size = grid_size
        self.y_space = numpy.linspace(rect['lon_min'], rect['lon_max'], grid_size)
        self.x_space = numpy.linspace(rect['lat_min'], rect['lat_max'], grid_size)
        self.regions, self.adj_matrix, self.degree_matrix = RegionGrid.create_regions(grid_size, self.x_space,
                                                                                      self.y_space)
        self.load_poi(poi)
        self.feature_matrix = self.create_feature_matrix()

    def load_poi(self, poi):
        x_space = self.x_space
        y_space = self.y_space
        regions = self.regions
        for id, poi_obj in poi.items():
            lat = poi_obj.location.lat
            long = poi_obj.location.lon
            # probably a better way
            # (tuple(array))
            # get the index of the bucket , we first find buckets in the space
            # that are less and pick the last one
            x_bucket = numpy.where(x_space == x_space[x_space <= lat][-1])[0][0]
            y_bucket = numpy.where(y_space == y_space[y_space <= long][-1])[0][0]

            regions[f"{x_bucket},{y_bucket}"].add_poi(poi_obj)

    def get_region_for_coor(self, lat, long):
        if lat < self.lat_min or lat > self.lat_max:
            return "Lat is out of X Coordinate Space"
        if long < self.lon_min or long > self.lon_max:
            return "Long is out of Y Coordinate Space"
        x_space = self.x_space
        y_space = self.y_space
        regions = self.regions
        x_bucket = x_space[x_space <= lat][-1]
        y_bucket = y_space[y_space <= long][-1]

        x_bucket_index = numpy.where(x_space == x_bucket)[0][0]
        y_bucket_index = numpy.where(y_space == y_bucket)[0][0]
        r_key = f"{x_bucket_index},{y_bucket_index}"
        if r_key in regions:
            return regions[r_key]
        return "No Region Found"

    @staticmethod
    def create_regions(grid_size, x_space, y_space):
        regions = {}
        grid_index = {}
        id = 0
        for x_point in range(0, grid_size):
            for y_point in range(0, grid_size):
                nw, ne, sw, se = None, None, None, None
                if x_point + 1 < grid_size and y_point + 1 < grid_size:
                    nw = (x_space[x_point], y_space[y_point])
                    ne = (x_space[x_point + 1], y_space[y_point])
                    sw = (x_space[x_point], y_space[y_point + 1])
                    se = (x_space[x_point + 1], y_space[y_point + 1])
                else:
                    if x_point + 1 < grid_size:
                        ne = (x_space[x_point + 1], y_space[y_point])
                    if y_point + 1 < grid_size:
                        sw = (x_space[x_point], y_space[y_point + 1])
                r = Region(f"{x_point},{y_point}", id, {'nw': nw, 'ne': ne, 'sw': sw, 'se': se})
                id += 1
                regions[f"{x_point},{y_point}"] = r
                for key in RegionGrid.key_gen(x_point, y_point):
                    if key in grid_index:
                        grid_index[key].append(r)
                    else:
                        grid_index[key] = [r]

        # adjacency matrix
        v = pow(grid_size, 2)
        # all zeroes
        matrix = numpy.zeros((v, v))
        # for regions, append touching regions
        for region_id, region in regions.items():
            id = region.id
            index = region.index.split(",")
            x = int(index[0])
            y = int(index[1])
            adj = {}
            for key in RegionGrid.key_gen(x, y):
                for region_from_index in grid_index[key]:
                    if region_from_index.index != region.index:
                        adj[str(region_from_index.index)] = region_from_index
            region.create_adjacency(adj)
            # for each adjacent region, get its id and set the column for it to 1 in the adj matrix
            for adj_region in adj.values():
                adj_index = adj_region.id
                matrix[id][adj_index] = 1
        return regions, matrix, RegionGrid.get_degree_mtx(matrix)

    @staticmethod
    def get_degree_mtx(A):
        return numpy.diag(numpy.sum(A, axis=1))

    @staticmethod
    def handle_poi(poi_data):
        poi_lon_min, poi_lon_max, poi_lat_min, poi_lat_max = None, None, None, None
        cat_to_index = {}
        for id, poi_obj in poi_data.items():
            lat = poi_obj.location.lat
            long = poi_obj.location.lon
            # category of poi data
            cat = poi_obj.cat
            if cat not in cat_to_index:
                current_size = len(cat_to_index)
                cat_to_index[cat] = current_size
            if poi_lon_min is None:
                poi_lon_min = long
                poi_lon_max = long
                poi_lat_min = lat
                poi_lat_max = lat
            else:
                if long < poi_lon_min:
                    poi_lon_min = long
                if long > poi_lon_max:
                    poi_lon_max = long
                if lat < poi_lat_min:
                    poi_lat_min = lat
                if lat > poi_lat_max:
                    poi_lat_max = lat
        rect = {
            "lon_min": poi_lon_min,
            "lon_max": poi_lon_max,
            "lat_min": poi_lat_min,
            "lat_max": poi_lat_max
        }
        return rect, cat_to_index

    @staticmethod
    def key_gen(x_point, y_point):
        return [
            f"{x_point}:{y_point}",
            f"{x_point}:{y_point + 1}",
            f"{x_point + 1}:{y_point}",
            f"{x_point + 1}:{y_point + 1}",
        ]

    def create_feature_matrix(self):
        regions = self.regions
        n = pow(self.grid_size, 2)
        m = len(self.categories)
        print(f"Creating Feature Matrix of size {n} X {m}")
        feature_matrix = numpy.zeros(shape=(n, m))
        for region in regions.values():
            index = region.id
            cats = region.categories
            for cat in cats:
                cat_index = self.categories[cat]
                current = feature_matrix[index][cat_index] + 1
                feature_matrix[index][cat_index] = current

        return feature_matrix


class Region:

    def __init__(self, index, id, points):
        self.id = id
        self.index = index
        self.points = points
        self.categories = set()
        self.nw, self.ne, self.sw, self.se = points['nw'], points['ne'], points['sw'], points['se']
        self.poi = []
        self.adjacent = {}
        self.move = self.move_keys()

    def move_keys(self):
        index = self.index.split(',')
        x = int(index[0])
        y = int(index[1])
        return {
            'n': f"{x},{y - 1}",
            'ne': f"{x + 1},{y - 1}",
            'e': f"{x + 1},{y}",
            'se': f"{x + 1},{y + 1}",
            's': f"{x},{y + 1}",
            'sw': f"{x - 1},{y + 1}",
            'w': f"{x - 1},{y}",
            'nw': f"{x - 1},{y - 1}"
        }

    def create_adjacency(self, regions):
        self.adjacent = regions

    def move_region_direction(self, dir):
        if dir in self.move:
            key = self.move[dir]
            if key in self.adjacent:
                return self.adjacent[key]
            else:
                return f"Cannot Move to {key} from {self.index}"
        else:
            return "Movement Not found"

    def add_poi(self, poi):
        self.poi.append(poi)
        self.categories.add(poi.cat)


if __name__ == '__main__':
    c = get_config()
    file = open(c["poi_file"], 'rb')
    region_grid = RegionGrid(file, 100)
    A = region_grid.adj_matrix
    D = region_grid.degree_matrix
    cat = region_grid.categories
    print(region_grid.feature_matrix[6671])
    print(numpy.nonzero(region_grid.feature_matrix[6671]))
    for cat in region_grid.regions['66,71'].categories:
        print(region_grid.categories[cat])
