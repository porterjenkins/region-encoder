import sys
import os
import numpy as np
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config import get_config
from grid.create_grid import RegionGrid
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
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

        errors = np.zeros(self.n)

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
            errors[i] = mse

        mean_cv_err = np.mean(errors)
        std_cv_err = np.std(errors)

        return mean_cv_err, std_cv_err, errors




if __name__ == "__main__":

    c = get_config()
    grid_size = 50
    file = open(c["poi_file"], 'rb')
    img_dir = c['path_to_image_dir']
    region_grid = RegionGrid(grid_size, poi_file=file, img_dir=img_dir, w_mtx_file=c['flow_mtx_file'],
                             housing_data=c["housing_data_file"], load_imgs=False)

    y_house = region_grid.get_target_var("house_price")
    y_is_valid = np.where(~np.isnan(y_house))[0]
    y_house = y_house[y_is_valid]

    #W = region_grid.weighted_mtx

    results = []

    # euclidean model
    D_euclidean = region_grid.get_distance_mtx()
    D_euclidean = D_euclidean[y_is_valid, :]
    D_euclidean = D_euclidean[:, y_is_valid]
    mod_euclidean = SimilarityModel(y_house, D_euclidean)

    err_euclidean, std_euclidean, _ = ols_euclidean = mod_euclidean.cv_ols()
    results.append(['euclidean', err_euclidean, std_euclidean])




    results = pd.DataFrame(results, columns=['model', 'mean cv error', 'std cv error'])
    print(results)


