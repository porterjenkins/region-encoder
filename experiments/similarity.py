import sys
import os
import numpy as np
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config import get_config
from grid.create_grid import RegionGrid
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error


class SimilarityModel(object):
    def __init__(self, y, weight_mtx):
        self.y = y
        self.weight_mtx = weight_mtx
        self.n = y.shape[0]

    def get_features(self, y, W):

        return np.dot(y, W)

    def cv_ols(self):
        indices = range(self.n)

        errors = np.zeros(self.n)

        for i in indices:

            train_idx = indices[:i] + indices[(i+1):]

            y_train = self.y[train_idx]
            W_train = self.weight_mtx[train_idx, train_idx]

            y_test = self.y[i]

            X_train = self.get_features(y_train, W_train)

            ols = LinearRegression()
            ols.fit(X=X_train, y=y_train)
            #ols.predict

            #errors[i] = mean_squared_error(y_true=y_test)










c = get_config()
grid_size = 5
file = open(c["poi_file"], 'rb')
img_dir = c['path_to_image_dir']
region_grid = RegionGrid(grid_size, poi_file=file, img_dir=img_dir, w_mtx_file=c['flow_mtx_file'],
                         housing_data=c["housing_data_file"], load_imgs=False)

y_house = region_grid.get_target_var("house_price")
y_is_valid = np.where(~np.isnan(y_house))[0]
y_house = y_house[y_is_valid]
D_euclidean = region_grid.get_distance_mtx()

W = region_grid.weighted_mtx

