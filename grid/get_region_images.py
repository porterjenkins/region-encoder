import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config import get_config
from grid.create_grid import RegionGrid, get_images_for_grid

c = get_config()
grid_size = 5
file = open(c["poi_file"], 'rb')
img_dir = c['path_to_image_dir']
region_grid = RegionGrid(grid_size,
                         poi_file=file,
                         lat_min=c['lat_min'],
                         lat_max=c['lat_max'],
                         lon_min=c['lon_min'],
                         lon_max=c['lon_max'])

get_images_for_grid(region_grid)