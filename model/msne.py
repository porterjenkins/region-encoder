from sklearn.decomposition import NMF
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config import get_config
from grid.create_grid import RegionGrid
from model.utils import write_embeddings
import numpy as np

c = get_config()
region_grid = RegionGrid(config=c)
n_categories = len(region_grid.categories)


X_poi_dist = np.zeros((region_grid.n_regions, n_categories**2))
# get intraregion poi-poi distance
cntr = 0
print("Getting Intra-region POI-POI distance networks")
for r in region_grid.regions.values():
    print("--> progress: {:.4f}".format(cntr/region_grid.n_regions), end='\r')
    x = r.get_poi_poi_dist(region_grid.categories)
    X_poi_dist[r.index, :] = x.flatten()
    cntr += 1

print("Getting Intra-region POI-POI mobility networks")
# get intra-region poi-poi mobility
f = c['raw_flow_file'].split(".csv")[0] + "-sampled.csv"
region_grid.get_taxi_trips(f)
