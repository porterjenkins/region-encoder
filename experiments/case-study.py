import sys
import os
import numpy as np
import pandas as pd
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config import get_config
from grid.create_grid import RegionGrid
from model.utils import load_embedding
from sklearn.neighbors import NearestNeighbors


def get_knn(X, k):
    nbrs = NearestNeighbors(n_neighbors=k, algorithm='ball_tree').fit(X)
    distances, indices = nbrs.kneighbors(X)

    return distances, indices

def interpolate_space(X, n_intervals):
    X_dim_min = np.min(X, axis=0)
    X_dim_max = np.max(X, axis=0)


    interval = (X_dim_max - X_dim_min) / n_intervals

    points = np.zeros((n_intervals, X.shape[1]))

    lin_step = X_dim_min
    for i in range(n_intervals):
        points[i, :] = lin_step + interval

        lin_step = points[i, :]

    return points



def get_neigbors_for_point(point, X, k):
    point = point.reshape(1, -1)
    H_aug = np.concatenate((point, X))
    aug_dist, aug_idx = get_knn(H_aug, 2)
    point_idx = aug_idx[0, 1]

    dist, indices = get_knn(X, k)

    return indices[point_idx, :]




c = get_config()
region_grid = RegionGrid(config=c)
k = 6

H = load_embedding(c['embedding_file'])
nbrs = NearestNeighbors(n_neighbors=k, algorithm='ball_tree').fit(H)
distances, indices = nbrs.kneighbors(H)

points = interpolate_space(H, 3)

for p in points:
    p_neighbors = get_neigbors_for_point(p, H, k)
    r_i = p_neighbors[0]
    r_i_coor = region_grid.idx_coor_map[r_i]
    print("Target Region: {} - {}:".format(r_i, r_i_coor))

    for i, neighbor in enumerate(p_neighbors):
        if i == 0:
            continue
        neighbor_coor = region_grid.idx_coor_map[neighbor]
        print("Neighbor: {} - {}".format(i, neighbor_coor))
