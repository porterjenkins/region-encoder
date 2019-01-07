import sys
import os
import numpy as np
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config import get_config
from grid.create_grid import RegionGrid
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
import pandas as pd


class SimilarityModel(object):
    def __init__(self, y, weight_mtx):
        self.y = y
        self.weight_mtx = weight_mtx
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

        return mean_cv_err[0], std_cv_err[0], mean_cv_err[1], std_cv_err[1]




if __name__ == "__main__":

    c = get_config()
    region_grid = RegionGrid(config=c, load_imgs=False)

    y_house = region_grid.get_target_var("house_price")
    y_is_valid = np.where(~np.isnan(y_house))[0]
    y_house = y_house[y_is_valid]

    results = []


    # euclidean model
    D_euclidean = region_grid.get_distance_mtx()
    D_euclidean = D_euclidean[y_is_valid, :]
    D_euclidean = D_euclidean[:, y_is_valid]
    mod_euclidean = SimilarityModel(y_house, D_euclidean)

    mse, mse_std, mae, mae_std  = mod_euclidean.cv_ols()
    results.append(['euclidean', mse, mse_std, mae, mae_std])

    # Run with Taxi flow as weighted edges
    W = region_grid.weighted_mtx
    W = W[y_is_valid, :]
    W = W[:, y_is_valid]

    mod_flow = SimilarityModel(y_house, W)

    mse, mse_std, mae, mae_std  = mod_flow.cv_ols()
    results.append(['flow', mse, mse_std, mae, mae_std])


    # Run with deepwalk as similarity measure
    deepwalk = region_grid.load_embedding(c['deepwalk_file'])
    W_deepwalk = np.matmul(deepwalk, np.transpose(deepwalk))
    W_deepwalk = W_deepwalk[y_is_valid, :]
    W_deepwalk = W_deepwalk[:, y_is_valid]

    mod_deepwalk = SimilarityModel(y_house, W_deepwalk)

    mse, mse_std, mae, mae_std  = mod_deepwalk.cv_ols()
    results.append(['deepwalk', mse, mse_std, mae, mae_std])


    # Run with RegionEncoder as similarity measure
    re_embed = region_grid.load_embedding(c['embedding_file'])
    W_re = np.matmul(re_embed, np.transpose(re_embed))
    W_re = W_re[y_is_valid, :]
    W_re = W_re[:, y_is_valid]

    mod_re = SimilarityModel(y_house, W_re)

    mse, mse_std, mae, mae_std  = mod_re.cv_ols()
    results.append(['proposed', mse, mse_std, mae, mae_std])


    results = pd.DataFrame(results, columns=['model', 'cv mse', 'std mse', 'cv mae', 'std mae'])
    print(results)



    results.to_csv("similarity-results.csv", index=False)