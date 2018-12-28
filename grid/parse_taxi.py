from config import get_config
from create_grid import RegionGrid

c = get_config()
file = open(c["poi_file"], 'rb')
region_grid = RegionGrid(file, 50)


W = region_grid.create_flow_matrix(c['raw_flow_file'])
