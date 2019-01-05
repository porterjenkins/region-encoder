import sys
import os
import pickle
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config import get_config
from create_grid import RegionGrid


c = get_config()
file = open(c["poi_file"], 'rb')
region_grid = RegionGrid(5, poi_file=file, lat_min=c['lat_min'], lat_max=c['lat_max'], lon_min=c['lon_min'],
                         lon_max=c['lon_max'])


W = region_grid.create_flow_matrix(c['raw_flow_file'])


with open(c['flow_mtx_file'], 'wb') as f:
    pickle.dump(W, f)
