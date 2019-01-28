import sys
import os
import pickle
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config import get_config
from grid.create_grid import RegionGrid


c = get_config()
# init region
region_grid = RegionGrid(config=c)


W = region_grid.create_flow_matrix(c['raw_flow_file'], region_name=c['city_name'], sample=True, p=.005)


with open(c['flow_mtx_file'], 'wb') as f:
    pickle.dump(W, f)
