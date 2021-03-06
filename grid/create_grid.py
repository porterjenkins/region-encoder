import logging
import os
import pickle
import random
import sys
# this should add files properly
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config import get_config
import numpy
import pandas
from geopy.distance import distance
from scipy import ndimage
from scipy.spatial.distance import euclidean
from shapely import geometry
import re

logging.basicConfig(filename='region.log', filemode='w', level=logging.INFO,
                    format='%(asctime)s %(message)s')



class RegionGrid:
    """
    Latitude (North/South):
        Latitude lines are a numerical way to measure how far north or south of the equator a place is located.
        The equator is the starting point for measuring latitude--that's why it's marked as 0 degrees latitude.
        The number of latitude degrees will be larger the further away from the equator the place is located,
        all the way up to 90 degrees latitude at the poles.
    Longitude (East/West):
        Longitude lines are a numerical way to show/measure how far a location is east or west of a universal vertical
        line called the Prime Meridian. This Prime Meridian line runs vertically, north and south, right over the
        British Royal Observatory in Greenwich England, from the North Pole to the South Pole.
        As the vertical starting point for longitude, the Prime Meridian is numbered 0 degrees longitude
    """

    def __init__(self, config, sample_prob=None, use_config_lat_lon=True):

        poi = self.get_poi_pickle(config['poi_file'])
        poi_rect, self.categories = RegionGrid.handle_poi(poi)

        if use_config_lat_lon:
            self.lon_min = config['lon_min']
            self.lon_max = config['lon_max']
            self.lat_min = config['lat_min']
            self.lat_max = config['lat_max']
        else:
            self.lon_min = poi_rect["lon_min"]
            self.lon_max = poi_rect["lon_max"]
            self.lat_min = poi_rect["lat_min"]
            self.lat_max = poi_rect["lat_max"]

        self.grid_size = config['grid_size']
        self.img_dir = config['path_to_image_dir']
        self.weight_mtx_fname = config['flow_mtx_file']
        self.img_tensor = numpy.array
        # define y space with longitude
        self.y_space = numpy.linspace(self.lon_min, self.lon_max, self.grid_size + 1)
        # define x space latitude
        self.x_space = numpy.linspace(self.lat_min, self.lat_max, self.grid_size + 1)
        # create regions, adjacency matrix, degree matrix, and image tensor
        self.regions, self.adj_matrix, self.degree_matrix, self.matrix_idx_map, grid_partition_map = \
            RegionGrid.create_regions(
                config['grid_size'],
                self.x_space,
                self.y_space,
                sample_prob=sample_prob
            )
        self.load_poi(poi)
        self.n_regions = len(self.regions)
        # Reverse mapping: index to coordinate name
        self.idx_coor_map = dict(zip(self.matrix_idx_map.values(), self.matrix_idx_map.keys()))
        self.feature_matrix = self.create_feature_matrix()

    def load_poi(self, poi):
        regions = self.regions
        for id, poi_obj in poi.items():
            lat = poi_obj.location.lat
            long = poi_obj.location.lon
            region_coor = self._map_to_region(lat_lon=(lat, long))
            # try and add poi data to corresponding region
            # if region is does not exist, due to random sampling, or out of bounds, skip poi data object
            try:
                regions[region_coor].add_poi(poi_obj)

            except KeyError:
                pass

    def get_region_for_coor(self, lat, long):
        if lat < self.lat_min or lat > self.lat_max:
            logging.warning(f"Lat {lat} is out of X Coordinate Space")
            return None
        if long < self.lon_min or long > self.lon_max:
            logging.warning(f"Long {long} is out of Y Coordinate Space")
            return None
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
        logging.warning("No Region Found")
        return None

    def get_reverse_categories(self):
        d = {}
        for k, v in self.categories.items():
            d[v] = k
        return d


    @staticmethod
    def update_arr_two_dim_sampled(arr, regions, grid_partition_map):
        """
        Update a symmetric, 2-d matrix of grid_size^2 x grid_size^2 to correct dimensions, after taking random sample
        of regions
        :param arr: (np.array) 2-d array
        :param regions: (dict) dictioary of sampled regions
        :param grid_partition_map: (dict) dictionary mapping full grid (grid_size^2 x grid_size^2) coordinate to integer
        values
        :return:
        """
        old_shape = arr.shape
        new_shape = list(old_shape)
        new_shape[0] = len(regions)
        new_shape[1] = len(regions)

        arr_new = numpy.zeros(new_shape)

        i = 0
        for coor_i, r_i in regions.items():
            j = 0
            for coor_j, r_j in regions.items():
                old_idx_i = grid_partition_map[coor_i]
                old_idx_j = grid_partition_map[coor_j]
                arr_new[i, j] = arr[old_idx_i, old_idx_j]
                j += 1
            i += 1

        return arr_new

    @staticmethod
    def parse_price(price):
        price = price.lower()
        if "m" in price:
            return float(price[:price.index("m")]) * 1000000
        return float(price)

    def get_taxi_trips(self, fname, city):
        df = pandas.read_csv(fname)

        if city == 'nyc':
            loc_cols = ['pickup_latitude', 'pickup_longitude', 'dropoff_latitude', 'dropoff_longitude']
        elif city == 'chicago':
            loc_cols = ['Pickup Centroid Latitude', 'Pickup Centroid Longitude', 'Dropoff Centroid Latitude', 'Dropoff Centroid Longitude',]
        else:
            raise NotImplementedError("city must be 'chicago' or 'nyc' ")


        print("Mapping taxi trips to regions")
        for idx, row in df.iterrows():
            print("--> progress: {:.4f}".format(idx / df.shape[0]), end='\r')
            if row['pickup_region'] == row['dropoff_region']:
                trip = row[loc_cols].values
                r_coor = row['pickup_region']
                r = self.regions[r_coor]
                r.add_trip(trip)

    def load_housing_data(self, fname):
        df = pandas.read_csv(fname)
        # df = df[['lat', 'lon', 'sold', 'sqft']]

        reg_coor = numpy.zeros(df.shape[0], dtype=object)
        missed = 0
        for index, row in df.iterrows():
            lat, lon, price, sqft = float(row.lat), float(row.lon), RegionGrid.parse_price(row.sold), float(row.sqft)
            region = self.get_region_for_coor(lat, lon)

            if region is not None:
                region.add_home(price / float(sqft))
                reg_coor[index] = region.coordinate_name
            else:
                reg_coor[index] = numpy.nan
                missed += 1
        print(f"{missed} rows of zillow data not loaded")
        df['region_coor'] = reg_coor
        return df

    def load_traffic_data(self, fname, city):
        if city == 'chicago':

            df = pandas.read_csv(fname)
            reg_coor = numpy.zeros(df.shape[0], dtype=object)
            missed = 0
            for road_id, row in df.iterrows():
                lat, lon, volume, date = float(row.Latitude), float(row.Longitude), row['Total Passing Vehicle Volume'], \
                                         row['Date of Count']
                region = self.get_region_for_coor(lat, lon)

                if region is not None:
                    region.add_traffic_volume(volume)
                    reg_coor[road_id] = region.coordinate_name
                else:
                    reg_coor[road_id] = numpy.nan
                    missed += 1

            df['region_coor'] = reg_coor

        elif city == 'nyc':
            df = pandas.read_csv(fname, index_col=0)
            df = df[~pandas.isnull(df.the_geom)].reset_index()
            traffic_cols = [x for x in df.columns if 'traffic_' in x]
            mean_traffic = df[traffic_cols].mean(axis=1)
            df['traffic'] = mean_traffic
            df.drop(traffic_cols, axis=1, inplace=True)


            roads = numpy.zeros(df.shape[0], dtype=object)
            reg_coor = numpy.zeros(df.shape[0], dtype=object)
            missed = 0

            for idx, row in df.iterrows():
                multilinestring = row.the_geom
                points = re.search('\(\((.*)\)\)', multilinestring).group(1).split(", ")
                point_list = []
                for p in points:
                    lon, lat = p.split(' ')
                    point_list.append((float(lat), float(lon)))
                line = geometry.LineString(point_list)
                roads[idx] = line
                centroid = line.centroid.xy
                region = self.get_region_for_coor(centroid[0][0], centroid[1][0])

                if region is not None:
                    #region.add_traffic_volume(volume)
                    reg_coor[idx] = region.coordinate_name
                else:
                    reg_coor[idx] = numpy.nan
                    missed += 1
            df['region_coor'] = reg_coor
            df = df[~pandas.isnull(df.region_coor)]

            #df = pandas.wide_to_long(df, stubnames='traffic', i = ['From', 'To'], j='hour', sep='_').reset_index()
            #df = gp.GeoDataFrame(df)
            #df['the_geom'] = gp.GeoSeries(roads)

            #df = df.set_geometry('the_geom')
            #print(df.head())
            #print(df.geometry.name)
            stop = 0
        else:
            raise NotImplementedError("city must be 'nyc' or 'chicago'")

        print(f"{missed} rows of traffic records not loaded")

        return df



    def get_checkin_counts(self, metric='sum'):
        """
        Compute total check-in county by region
        :return: (np.array) array of counts
        """

        checkins = numpy.zeros((self.n_regions, 3))

        for coor, r, in self.regions.items():
            checkins[r.index, 0] = r.count_checkins(metric)
            checkins[r.index, 1] = r.mid_point[0]
            checkins[r.index, 2] = r.mid_point[1]

        return pandas.DataFrame(checkins, columns=['checkins', 'lat', 'lon'])


    def load_img_data(self, img_dims=(50,50), std_img=True):

        idx_cntr = 0
        # init image tensor: n_samples x n_channels x n_rows x n_cols
        self.img_tensor = numpy.zeros((self.grid_size ** 2, 3, img_dims[0], img_dims[1]), dtype=numpy.float32)
        for coor_idx, r in self.regions.items():
            r.load_sat_img(self.img_dir, standarize=std_img)
            self.img_tensor[idx_cntr, :, :, :] = r.sat_img

            idx_cntr += 1

    def load_weighted_mtx(self):
        self.weighted_mtx = self.get_mtx(self.weight_mtx_fname)
        self.weighted_mtx = RegionGrid.normalize_mtx(self.weighted_mtx)
        # if sample_prob is not None:
        #    # Update weighted matrix to reflect sampled regions
        #    self.weighted_mtx = RegionGrid.update_arr_two_dim_sampled(self.weighted_mtx, self.regions,
        #                                                              grid_partition_map)

    @staticmethod
    def create_regions(grid_size, x_space, y_space, sample_prob):
        logging.info("Running create regions job")
        regions = {}
        grid_index = {}
        index = 0
        # init image tensor: n_samples x n_channels x n_rows x n_cols
        # img_tensor = numpy.zeros((grid_size ** 2, 3, img_dims[0], img_dims[1]), dtype=numpy.float32)

        # Probability of sampling constructing a given region
        if sample_prob is not None:
            alpha = sample_prob
        else:
            alpha = 1.0

        grid_partition_map = {}
        grid_partition_cntr = 0

        for x_point in range(0, grid_size):
            for y_point in range(0, grid_size):

                if random.random() < alpha:
                    nw, ne, sw, se = None, None, None, None
                    nw = (x_space[x_point], y_space[y_point])
                    ne = (x_space[x_point + 1], y_space[y_point])
                    sw = (x_space[x_point], y_space[y_point + 1])
                    se = (x_space[x_point + 1], y_space[y_point + 1])
                    r = Region(f"{x_point},{y_point}", index, {'nw': nw, 'ne': ne, 'sw': sw, 'se': se})
                    # if load_imgs:
                    #    r.load_sat_img(img_dir, standarize=std_img)
                    #    img_tensor[index, :, :, :] = r.sat_img
                    print("Initializing region: %s" % r.coordinate_name, end='\r')
                    index += 1
                    regions[f"{x_point},{y_point}"] = r
                    for key in RegionGrid.key_gen(x_point, y_point):
                        if key in grid_index:
                            grid_index[key].append(r)
                        else:
                            grid_index[key] = [r]

                    grid_partition_map[f"{x_point},{y_point}"] = grid_partition_cntr
                    grid_partition_cntr += 1


                else:
                    # only increment counter
                    grid_partition_map[f"{x_point},{y_point}"] = grid_partition_cntr
                    grid_partition_cntr += 1

        # adjacency matrix
        v = len(regions)
        # all zeroes
        matrix = numpy.zeros((v, v), dtype=numpy.int64)
        # mapping of region coordinate to index
        coor_index_mapping = dict()
        # for regions, append touching regions
        for region_id, region in regions.items():
            # name to index
            coor_index_mapping[region.coordinate_name] = region.index

            index = region.index
            # split name
            name = region.coordinate_name.split(",")
            x = int(name[0])
            y = int(name[1])

            adj = {}
            # find all points touching these
            for key in RegionGrid.key_gen(x, y):
                for region_from_index in grid_index[key]:
                    if region_from_index.coordinate_name != region.coordinate_name:
                        adj[str(region_from_index.coordinate_name)] = region_from_index
            region.create_adjacency(adj)
            # for each adjacent region, get its id and set the column for it to 1 in the adj matrix
            for adj_region in adj.values():
                adj_index = adj_region.index
                matrix[index][adj_index] = 1

        # Update image tensor to correct dimensions, if sampling is used
        # if sample_prob is not None:
        #    img_tensor = img_tensor[range(v), :, :, :]

        return regions, matrix, RegionGrid.get_degree_mtx(matrix), coor_index_mapping, grid_partition_map

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
        n = self.n_regions
        m = len(self.categories)
        print(f"Creating Feature Matrix of size {n} X {m}")
        feature_matrix = numpy.zeros(shape=(n, m))
        for region in regions.values():
            index = region.index
            cats = region.categories
            for cat in cats:
                cat_index = self.categories[cat]
                current = feature_matrix[index][cat_index] + 1
                feature_matrix[index][cat_index] = current

        return feature_matrix

    def _cast_coor(self, lat_lon):
        """
        Cast the elements of a tuple (lat_lon) to float
        :param lat_lon: (tuple) latitude, longitude pair
        :return: (tuple) latitude, longitude pair as float
        """
        lat = float(lat_lon[0])
        lon = float(lat_lon[1])

        return (lat, lon)

    def _iscomplete(self, lat_lon):
        """
        Function to identify complete coordinate data for both drop-off and pickup
        :param lat_lon: (tuple) latitude, longitude pair
        :return: bool
        """

        if lat_lon[0] != '' and lat_lon[1] != '':
            return True
        else:
            return False

    def _map_to_region(self, lat_lon):
        """
        Map a given lat/lon pair to correct region
            - This function uses a quick constant time O(1) algorithm to map trip to region
        :param lat_lon: (tuple) latitude, longitude pair
        :return: (str) x, y indices of correct region as a csv
        """
        x = lat_lon[0]
        y = lat_lon[1]

        width = self.x_space[1] - self.x_space[0]
        heigth = self.y_space[1] - self.y_space[0]

        x_idx = int((x - self.x_space[0]) / width)
        y_idx = int((y - self.y_space[0]) / heigth)

        return "{},{}".format(x_idx, y_idx)

    def create_flow_matrix(self, fname, n_rows=None, region_name='chicago', sample=False, p=.05):
        """
        Generated a weighted matrix (dims: n_regions x n_regions) from taxi flow data
         (https://data.cityofchicago.org/Transportation/Taxi-Trips/wrvz-psew)
            - raw data has approx 99M rows

        :param fname: (str) file name to query for taxi flow
        :param n_rows: (int) optional: only take first n rows from file
        :return: (np.array) 2-d weighted flow matrix
        """
        # approx 99M rows
        # row_total = 99e6

        flow_matrix = numpy.zeros((self.n_regions, self.n_regions))
        # index given by chicago data portal docs
        if region_name == 'chicago':
            drop_lat_idx = 20
            drop_lon_idx = 21
            pickup_lat_idx = 17
            pickup_lon_idx = 18
        elif region_name == 'nyc':
            drop_lat_idx = 10
            drop_lon_idx = 9
            pickup_lat_idx = 6
            pickup_lon_idx = 5
        else:
            raise NotImplementedError("Taxi trajectory parsing only implemented for 'chicago' and 'nyc'")


        if sample:
            sample_fname = "{}-{}".format(fname.split(".csv")[0], 'sampled.csv')
            if os.path.exists(sample_fname):
                os.remove(sample_fname)



        sample_cnt = 0
        row_cntr = 0
        with open(fname, 'r') as f:
            for row in f:
                data = row.strip().split(",")

                if row_cntr == 0:
                    headers = data
                    if sample:
                        headers += ['pickup_region', 'dropoff_region']
                        with open(sample_fname, 'a') as f:
                            for item in headers:
                                f.write("%s," % item)
                            f.write("\n")

                else:
                    try:
                        trip_pickup = (data[pickup_lat_idx], data[pickup_lon_idx])
                        trip_drop = (data[drop_lat_idx], data[drop_lon_idx])
                    except IndexError:
                        continue

                    if self._iscomplete(trip_pickup) and self._iscomplete(trip_drop):
                        try:
                            trip_pickup = self._cast_coor(trip_pickup)
                            trip_drop = self._cast_coor(trip_drop)

                            pickup_region = self._map_to_region(trip_pickup)
                            drop_region = self._map_to_region(trip_drop)

                            p_idx = self.matrix_idx_map[pickup_region]
                            d_idx = self.matrix_idx_map[drop_region]

                            flow_matrix[p_idx, d_idx] += 1.0
                            sample_cnt += 1

                            if sample_cnt % 100 == 0:
                                print("{}, {} --> {}".format(sample_cnt, trip_pickup, trip_drop),
                                      end="\r")

                            if sample:
                                alpha = numpy.random.uniform(0, 1)
                                if alpha < p:
                                    with open(sample_fname, 'a') as f:
                                        pickup_region = '"%s"' % pickup_region
                                        drop_region = '"%s"' % drop_region
                                        data += [pickup_region, drop_region]
                                        for item in data:
                                            f.write("%s," % item)
                                        f.write("\n")


                            if n_rows is not None:
                                if sample_cnt >= n_rows:
                                    break

                        except (ValueError, KeyError) as err:
                            pass

                row_cntr += 1

        return flow_matrix

    def get_mtx(self, fname):
        with open(fname, 'rb') as f:
            W = pickle.load(f)
        return W

    def get_poi_pickle(self, fname):
        with open(fname, 'rb') as poi_f:
            poi = pickle.load(poi_f)
        return poi

    @staticmethod
    def arr_size(arr):
        b = arr.nbytes
        gb = b / 1000000000
        print("array size: {} GB".format(gb))
        return gb

    def get_target_var(self, target_name):

        y = numpy.zeros(self.n_regions)

        if target_name == 'house_price':
            n_nan = 0
            for id, r in self.regions.items():
                idx = self.matrix_idx_map[id]
                val = r.median_home_value()
                if numpy.isnan(val):
                    n_nan += 1
                y[idx] = val

            print("Pct of regions with missing zillow prices: {}".format(n_nan / (self.n_regions)))
            return y
        else:
            raise NotImplementedError("Only 'house_price' is currently implemented")

    def get_distance_mtx(self, metric='euclidean'):
        """
        Get pairwise distance matrix of all regions. Default metric is euclidean distance between
            region_i and region_j
        First an upper-triangular matrix for effeciency, then transpose and copy to get full symmetric matrix

        :param metric:
        :return: (np.array) Upper-triangular matrix of spatial distances
        """
        print("Creating distance matrix -- metric = {}".format(metric))

        dist_mtx = numpy.zeros((self.n_regions, self.n_regions))

        # iterate over regions
        # create upper-triangular matrix for efficiency
        for i, r_i in self.regions.items():
            idx_i = self.matrix_idx_map[i]
            for idx_j in range(idx_i + 1, self.n_regions):
                j = self.idx_coor_map[idx_j]
                r_j = self.regions[j]

                print("progress -- i: {}, j: {}".format(i, j), end='\r')
                if metric == 'euclidean':
                    try:
                        dist = euclidean(r_i.mid_point, r_j.mid_point)
                    except ValueError:
                        dist = numpy.nan
                else:
                    raise NotImplementedError("Only metric='euclidean' is currently implemented")

                dist_mtx[idx_i, idx_j] = dist

        # Copy upper trianguler matrix to lower triangular matrix and add to distance matrix
        dist_mtx = dist_mtx + numpy.transpose(dist_mtx)

        return dist_mtx

    @staticmethod
    def normalize_mtx(mtx):
        row_sums = numpy.sum(mtx, axis=1).reshape(-1, 1)
        mtx_norm = mtx / row_sums
        idx_nan = numpy.isnan(mtx_norm)
        mtx_norm[idx_nan] = 0.0
        return mtx_norm

    def write_adj_list(self, fname):
        """
        Write adjacency list to file using adjacency matrix
        :param fname:
        :return: None
        """
        with open(fname, 'w') as f:
            for i, row in enumerate(self.adj_matrix):
                f.write(str(i) + " ")
                adj_list = numpy.where(row > 0.0)[0].astype(numpy.int32)
                for j, neighbor in enumerate(adj_list):
                    if j == len(adj_list) - 1:
                        f.write(str(neighbor))
                    else:
                        f.write(str(neighbor) + " ")
                f.write("\n")

    def create_triplets(self, indices, tensor):
        n, d = [], []
        for idx in indices:
            # random neighbor
            n.append(numpy.random.choice(numpy.nonzero(self.adj_matrix[idx])[0]))
            # random distant
            d.append(numpy.random.choice(numpy.where(self.adj_matrix == 0)[0]))

        patch = tensor[indices, :, :, :]
        n = tensor[n, :, :, :]
        d = tensor[d, :, :, :]
        return patch, n, d

    def write_edge_list(self, fname):

        with open(fname, 'w') as f:
            for i, row in enumerate(self.adj_matrix):
                adj_list = numpy.where(row > 0.0)[0].astype(numpy.int32)
                for j, neighbor in enumerate(adj_list):
                    f.write("{} {}\n".format(str(i), str(neighbor)))



class Region:

    def __init__(self, name, index, points):
        self.index = index
        self.coordinate_name = name
        self.points = points
        self.categories = set()
        self.nw, self.ne, self.sw, self.se = points['nw'], points['ne'], points['sw'], points['se']
        self.mid_point = self.compute_midpoint()
        self.poi = []
        self.adjacent = {}
        self.move = self.move_keys()
        self.sat_img = numpy.array
        self.home_data = []
        self.traffic_data = []
        self.trips = []

    def median_home_value(self):
        return numpy.median(self.home_data)

    def mean_home_value(self):
        return numpy.mean(self.home_data)

    def max_home_value(self):
        return numpy.max(self.home_data)

    def min_home_value(self):
        return numpy.min(self.home_data)

    def add_home(self, home):
        self.home_data.append(home)

    def add_trip(self, trip):
        self.trips.append(trip)

    def add_traffic_volume(self, volume):
        self.traffic_data.append(volume)

    def move_keys(self):
        index = self.coordinate_name.split(',')
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

    def count_checkins(self, metric=False):

        #total_cnt = 0
        #poi_cnt = len(self.poi)
        poi_cnts = list()
        if self.poi:
            for p in self.poi:
                #total_cnt += p.checkin_count
                poi_cnts.append(p.checkin_count)
        else:
            poi_cnts.append(0)

        if metric == 'sum':
            return numpy.sum(poi_cnts)
        elif metric == 'mean':
            return numpy.mean(poi_cnts)
        elif metric == 'median':
            return numpy.median(poi_cnts)
        else:
            raise NotImplementedError("{} not implemented as poi checkin metric".format(metric))

    def create_adjacency(self, regions):
        self.adjacent = regions

    def move_region_direction(self, dir):
        if dir in self.move:
            key = self.move[dir]
            if key in self.adjacent:
                return self.adjacent[key]
            else:
                return f"Cannot Move to {key} from {self.coordinate_name}"
        else:
            return "Movement Not found"

    def add_poi(self, poi):
        self.poi.append(poi)
        self.categories.add(poi.cat)

    def load_sat_img(self, img_dir, standarize):
        coors_split = self.coordinate_name.split(",")
        coors = "-".join(coors_split)
        fname = "{}/{}.png".format(img_dir, coors)
        img = ndimage.imread(fname)
        img_t = numpy.transpose(img).astype(numpy.float32)

        if standarize:
            for channel in range(img_t.shape[0]):
                # standardize each channel --> zero-mean, unit variance
                mu = numpy.mean(img_t[channel, :, :])
                sig = numpy.std(img_t[channel, :, :])
                img_t[channel, :, :] = (img_t[channel, :, :] - mu) / sig

        # ensure that each image has three change (r, g, b)
        if img_t.shape[0] > 3:
            logging.info('Image for region {} has shape {}. Using first three on dim 0'.format(coors, img_t.shape))
            self.sat_img = img_t[0:3, :, :]
        else:
            self.sat_img = img_t

    def compute_distances(self):

        x_points = (self.points['nw'], self.points['sw'])
        y_points = (self.points['nw'], self.points['ne'])

        x_dist = distance(x_points[0], x_points[1]).km
        y_dist = distance(y_points[0], y_points[1]).km

        return x_dist, y_dist

    def compute_midpoint(self):
        """
        Compute the dimensions of a region in Kilometers
        :return:
        """

        try:
            x_points = [self.points['nw'][0], self.points['ne'][0]]
            y_points = [self.points['nw'][1], self.points['sw'][1]]

            x_mid = numpy.mean(x_points)
            y_mid = numpy.mean(y_points)
            mid = [x_mid, y_mid]
        except TypeError:
            mid = numpy.nan

        return mid

    def get_poi_poi_dist(self, cat_idx_map):
        """
        Get the POI-to-POI category istance matrix (network).
            each element, ij, denotes the average distance in meters from POI category i to j
        :param cat_idx_map: (dict) mapping from POI categories to matrix indices
        :return: poi_poi_dist (np.array)
        """
        n_cat = len(cat_idx_map)

        category_sum_mtx = numpy.zeros((n_cat, n_cat))
        category_cnt_mtx = numpy.zeros((n_cat, n_cat))

        for i, poi_i in enumerate(self.poi):
            for j, poi_j in enumerate(self.poi):
                point_i = (poi_i.location.lat, poi_i.location.lon)
                point_j = (poi_j.location.lat, poi_j.location.lon)

                idx_i = cat_idx_map[poi_i.cat]
                idx_j = cat_idx_map[poi_j.cat]
                d = distance(point_i, point_j).meters

                category_sum_mtx[idx_i, idx_j] += d
                category_cnt_mtx[idx_i, idx_j] += 1

        poi_poi_dist = category_sum_mtx / category_cnt_mtx

        return numpy.nan_to_num(poi_poi_dist, 0)

    def get_poi_poi_mobility(self):
        pass


def get_images_for_grid(region_grid, clear_dir=False, compress=True):
    from image.image_retrieval import get_images_for_all_no_marker, compress_images
    get_images_for_all_no_marker(region_grid, clear_dir=clear_dir)
    if compress:
        compress_images(region_grid.img_dir, resize=(50,50))


if __name__ == '__main__':
    c = get_config()
    region_grid = RegionGrid(config=c)
    tmp = region_grid.feature_matrix.sum(axis=1)

    checkins = region_grid.get_checkin_counts()
    for r in region_grid.regions.values():
        print(r.compute_distances())

    # region_grid.load_img_data(std_img=True)
    region_grid.load_weighted_mtx()
    region_grid.load_housing_data(c['housing_data_file'])

    A = region_grid.adj_matrix
    D = region_grid.degree_matrix
    W = region_grid.weighted_mtx

    # I = region_grid.img_tensor

    print(W.shape)
    print(W.sum())
    print(A.shape)
    print(D.shape)
    # print(I.shape)
    y_house = region_grid.get_target_var("house_price")
    print(y_house)

    # print(I)

