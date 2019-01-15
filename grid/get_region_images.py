import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config import get_config
from grid.create_grid import RegionGrid, get_images_for_grid

c = get_config()
region_grid = RegionGrid(config=c)

get_images_for_grid(region_grid, clear_dir=True)