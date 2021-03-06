import sys
import os
import numpy as np
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config import get_config
from grid.create_grid import RegionGrid
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from model.utils import load_embedding

def get_knn(X, k, idx_coor_map):
    nbrs = NearestNeighbors(n_neighbors=k, algorithm='ball_tree')
    nbrs.fit(X)
    distances, indices = nbrs.kneighbors(X)
    for i in range(X.shape[0]):
        id_i = idx_coor_map[i]
        print("{}: region {}: ".format(i, id_i))
        nbhrs_id = list()
        for j in range(k):
            if j > 0:
                mtx_idx = indices[i, j]
                nbhrs_id.append(idx_coor_map[mtx_idx])
        nbhrs_str = ", ".join(nbhrs_id)
        print("--> " + nbhrs_str)

    return distances, indices


def cv_adj_mean(regions, matrix_idx_map, y):
    n = len(y)
    errors = [[], []]

    region_cntr = 0
    for coor, r in regions.items():
        adj_list = r.adjacent

        r_mtx_idx = matrix_idx_map[coor]
        y_test = [y[r_mtx_idx]]
        if np.isnan(y_test):
            continue
        y_hat = 0
        is_valid_y_cnt = 0

        for neighbor_coor in adj_list:
            n_mtx_id = matrix_idx_map[neighbor_coor]

            if np.isnan(y[n_mtx_id]):
                continue
            else:
                y_hat += y[n_mtx_id]
                is_valid_y_cnt += 1

        y_hat = [y_hat / is_valid_y_cnt]
        mse = mean_squared_error(y_true=y_test, y_pred=y_hat)
        mae = mean_absolute_error(y_test, y_hat)
        errors[0].append(mse)
        errors[1].append(mae)

        region_cntr += 1
    errors = np.transpose(np.array(errors))
    mean_cv_err = np.round(np.mean(errors, axis=0), 4)
    std_cv_err = np.round(np.std(errors, axis=0), 4)

    return mean_cv_err[0], std_cv_err[0], mean_cv_err[1], std_cv_err[1], errors


def cv_naive_mean(y):
    n = len(y)
    indices = list(range(n))

    errors = np.zeros((n, 2))

    for i in indices:
        train_idx = indices[:i] + indices[(i + 1):]
        y_train = y[train_idx]
        y_hat = [np.mean(y_train)]

        y_test = [y[i]]

        mse = mean_squared_error(y_true=y_test, y_pred=y_hat)
        mae = mean_absolute_error(y_test, y_hat)
        errors[i, 0] = mse
        errors[i, 1] = mae

    mean_cv_err = np.round(np.mean(errors, axis=0), 4)
    std_cv_err = np.round(np.std(errors, axis=0), 4)

    return mean_cv_err[0], std_cv_err[0], mean_cv_err[1], std_cv_err[1], errors

class SimilarityModel(object):
    def __init__(self, y, weight_mtx=None):
        self.y = y
        self.weight_mtx = weight_mtx
        if weight_mtx is not None:
            self.spatial_features = self.get_features(y, weight_mtx)
        self.n = y.shape[0]

    def get_features(self, y, W):

        return np.dot(y, W).reshape(-1, 1)


    def cv_ols(self):
        indices = list(range(self.n))

        errors = np.zeros((self.n, 2))

        for i in indices:

            train_idx = indices[:i] + indices[(i+1):]

            y_train = self.y[train_idx]
            X_train = self.spatial_features[train_idx, :]

            ols = LinearRegression()
            ols.fit(X=X_train, y=y_train)

            y_test = [self.y[i]]
            X_test = self.spatial_features[i, :].reshape(-1, 1)
            y_hat = ols.predict(X_test)

            mse = mean_squared_error(y_true=y_test, y_pred=y_hat)
            mae = mean_absolute_error(y_test, y_hat)
            errors[i, 0] = mse
            errors[i, 1] = mae

        mean_cv_err = np.round(np.mean(errors, axis=0), 4)
        std_cv_err = np.round(np.std(errors, axis=0), 4)

        return mean_cv_err[0], std_cv_err[0], mean_cv_err[1], std_cv_err[1], errors




if __name__ == "__main__":

    c = get_config()
    region_grid = RegionGrid(config=c)
    region_grid.load_weighted_mtx()
    region_grid.load_housing_data(c['housing_data_file'])

    y_house = region_grid.get_target_var("house_price")
    results = []


    # Adjacency Average Model
    mse, mse_std, mae, mae_std, err_adj = cv_adj_mean(region_grid.regions, region_grid.matrix_idx_map, y_house)
    results.append(['adjacent avg', mse, mse_std, mae, mae_std])

    y_is_valid = np.where(~np.isnan(y_house))[0]
    y_house = y_house[y_is_valid]



    # Global Avg Model
    global_avg = SimilarityModel(y_house)
    mse, mse_std, mae, mae_std, err_euclidean = cv_naive_mean(y_house)
    results.append(['global avg', mse, mse_std, mae, mae_std])


    # euclidean model
    D_euclidean = region_grid.get_distance_mtx()
    D_euclidean = D_euclidean[y_is_valid, :]
    D_euclidean = D_euclidean[:, y_is_valid]
    mod_euclidean = SimilarityModel(y_house, D_euclidean)

    mse, mse_std, mae, mae_std, err_euclidean = mod_euclidean.cv_ols()
    results.append(['euclidean', mse, mse_std, mae, mae_std])

    # Run with Taxi flow as weighted edges
    W = region_grid.weighted_mtx
    W = W[y_is_valid, :]
    W = W[:, y_is_valid]

    mod_flow = SimilarityModel(y_house, W)

    mse, mse_std, mae, mae_std, err_flow = mod_flow.cv_ols()
    results.append(['flow', mse, mse_std, mae, mae_std])

    # Matrix Factorization as similarity
    nmf = load_embedding(c['nmf_file'])
    W_nmf = np.matmul(nmf, np.transpose(nmf))
    W_nmf = W_nmf[y_is_valid, :]
    W_nmf = W_nmf[:, y_is_valid]

    mod_nmf = SimilarityModel(y_house, W_nmf)
    mse, mse_std, mae, mae_std, err_nmf = mod_nmf.cv_ols()
    results.append(['matrix factorization', mse, mse_std, mae, mae_std])


    # Run with deepwalk as similarity measure
    deepwalk = load_embedding(c['deepwalk_file'])
    W_deepwalk = np.matmul(deepwalk, np.transpose(deepwalk))
    W_deepwalk = W_deepwalk[y_is_valid, :]
    W_deepwalk = W_deepwalk[:, y_is_valid]

    mod_deepwalk = SimilarityModel(y_house, W_deepwalk)

    mse, mse_std, mae, mae_std, err_deepwalk = mod_deepwalk.cv_ols()
    results.append(['deepwalk', mse, mse_std, mae, mae_std])


    # Run with RegionEncoder as similarity measure
    re_embed = load_embedding(c['embedding_file'])
    W_re = np.matmul(re_embed, np.transpose(re_embed))
    W_re = W_re[y_is_valid, :]
    W_re = W_re[:, y_is_valid]

    mod_re = SimilarityModel(y_house, W_re)

    mse, mse_std, mae, mae_std, err_re = mod_re.cv_ols()
    results.append(['proposed', mse, mse_std, mae, mae_std])


    results = pd.DataFrame(results, columns=['model', 'cv mse', 'std mse', 'cv mae', 'std mae'])
    print(results)



    results.to_csv("similarity-results.csv", index=False)




    ## Post hoc analysis of neighborhoods
    #print("---- KNN Analysis: Euclidean -----")
    #get_knn(D_euclidean, 5, region_grid.idx_coor_map)
    #print("---- KNN Analysis: DeepWalk -----")
    #get_knn(deepwalk, 5, region_grid.idx_coor_map)
    #print("---- KNN Analysis: Proposed -----")
    #get_knn(re_embed, 5, region_grid.idx_coor_map)
