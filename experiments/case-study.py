import sys
import os
import numpy as np
import pandas as pd
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config import get_config
from grid.create_grid import RegionGrid
from model.utils import load_embedding
from sklearn.neighbors import NearestNeighbors
from sklearn.manifold import TSNE


def get_knn(X, k):
    nbrs = NearestNeighbors(n_neighbors=k, algorithm='ball_tree').fit(X)
    distances, indices = nbrs.kneighbors(X)

    return distances, indices

def interpolate_space(X, n_intervals):

    tsne = TSNE(n_components=2)
    X_2d = tsne.fit_transform(X)

    dim_min = np.min(X_2d, axis=0)
    dim_max = np.max(X_2d, axis=0)

    change = dim_max - dim_min
    slope = change[1] / change[0]

    x_steps = np.linspace(dim_min[0], dim_max[0], n_intervals)

    points = np.zeros((n_intervals, 2))
    for i, x in enumerate(x_steps):
        y = slope*x

        points[i, :] = [x, y]



    return points, X_2d


def get_neigbors_for_point(point, X, k):
    point = point.reshape(1, -1)
    H_aug = np.concatenate((point, X))
    aug_dist, aug_idx = get_knn(H_aug, k)
    point_idx = aug_idx[0, 1]

    dist, indices = get_knn(X, k)

    return indices[point_idx, :]

def count_poi(region):
    poi_dict = {}
    for poi in region.poi:
        if poi.cat == '':
            poi_cat = 'None'
        else:
            poi_cat = poi.cat

        if poi_cat in poi_dict:
            poi_dict[poi_cat] += 1
        else:
            poi_dict[poi_cat] =1

    return poi_dict


def filter_dict(d, top_k):
    d_new = {}
    sorted_by_value = sorted(d.items(), key=lambda kv: kv[1], reverse=True)
    for item in sorted_by_value[:top_k]:
        d_new[item[0]] = item[1]

    return d_new

c = get_config()
region_grid = RegionGrid(config=c)
k = 5

H = load_embedding(c['embedding_file'])
X = region_grid.feature_matrix
nbrs = NearestNeighbors(n_neighbors=k, algorithm='ball_tree').fit(H)
distances, indices = nbrs.kneighbors(H)


query = ['37,22','33,21','48,12','47,23']

points = [region_grid.matrix_idx_map[q] for q in query]


#points, H_2d = interpolate_space(H, 4)


for p in points:
    result = {}
    #p_neighbors = get_neigbors_for_point(p, H_2d, k)
    p_neighbors = indices[p, :]
    r_i_id = p_neighbors[0]
    r_i_coor = region_grid.idx_coor_map[r_i_id]
    print("Target Region: {} - {}:".format(r_i_id, r_i_coor))
    print("------------------")
    r_i = region_grid.regions[r_i_coor]
    poi_dict = count_poi(r_i)
    poi_filter = filter_dict(poi_dict, 5)
    result['target'] = poi_filter


    for i, neighbor in enumerate(p_neighbors):
        if i == 0:
            continue
        neighbor_coor = region_grid.idx_coor_map[neighbor]
        print("Neighbor: {} - {}".format(i, neighbor_coor))
        print("------------------")
        r_j = region_grid.regions[neighbor_coor]
        nbr_poi_dict = count_poi(r_j)
        key = "neighbor_{}".format(i+1)
        r_j_poi_filter = filter_dict(nbr_poi_dict, 5)
        result[key] = r_j_poi_filter


    import pandas as pd
    df = pd.DataFrame.from_dict(result, orient='index').fillna(0)
    df = df[sorted(df.columns)]

    df.to_csv("results/{}-poi.csv".format(r_i_coor))
