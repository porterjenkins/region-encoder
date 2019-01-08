import sys
import os
import pandas as pd
import numpy as np
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config import get_config
from grid.create_grid import RegionGrid
import xgboost



c = get_config()
zillow = pd.read_csv(c['housing_data_file'])

print(zillow.head())
print(zillow.shape)

region_grid = RegionGrid(config=c, load_imgs=False)
z = region_grid.load_housing_data(c['housing_data_file'])




