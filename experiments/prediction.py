import sys
import os
import pandas as pd
import numpy as np
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import xgboost
from sklearn.metrics import mean_squared_error, mean_absolute_error
from model.utils import load_embedding
from sklearn.linear_model import Lasso
from sklearn.ensemble import RandomForestRegressor


class PredictionModel(object):
    def __init__(self, idx_coor_map, config, n_epochs, embedding=None, second_embedding=None):
        self.idx_coor_map = idx_coor_map
        self.config = config
        self.n_epochs = n_epochs
        self.embedding = embedding
        self.second_embedding = second_embedding
        self.param = {
            'objective': 'reg:linear',
            'eta': 0.02,
            'eval_metric': 'rmse',
            'max_depth': 5,
            'min_child_weight': 3,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'silent': 1,
            'seed': 123
        }

        self.X = np.array
        self.y = np.array

    def __is_fname(self, arr_or_fname):
        if isinstance(arr_or_fname, str):
            embed = load_embedding(arr_or_fname)
        else:
            embed = arr_or_fname

        return embed

    def get_embedding(self):

        if self.embedding is not None:
            embed = self.__is_fname(self.embedding)
            if self.second_embedding is not None:
                second_embed = self.__is_fname(self.second_embedding)
                embed = np.concatenate((embed, second_embed), axis=1)

            return embed
        else:
            return None


    def get_features(self, input_data):
        pass


    def train_eval(self, train_idx, test_idx, model='xgb'):


        if model == 'xgb':
            trn = xgboost.DMatrix(self.X[train_idx, :], label=self.y[train_idx])
            tst = xgboost.DMatrix(self.X[test_idx, :], label=self.y[test_idx])

            eval_list = [(trn, 'train')]
            model = xgboost.train(self.param, trn, self.n_epochs, verbose_eval=True, evals=eval_list)
            pred = model.predict(tst)
        elif model == 'lasso':

            model = Lasso(alpha=1.0, fit_intercept=True)
            model.fit(X=self.X[train_idx, :], y=self.y[train_idx])
            pred = model.predict(X=self.X[test_idx])

        elif model == 'rf':

            model = RandomForestRegressor()
            model.fit(X=self.X[train_idx, :], y=self.y[train_idx])
            pred = model.predict(X=self.X[test_idx])



        rmse = np.sqrt(mean_squared_error(self.y[test_idx], pred))
        mae = mean_absolute_error(self.y[test_idx], pred)

        return rmse, mae

class HousePriceModel(PredictionModel):
    def __init__(self, idx_coor_map, config, n_epochs, embedding=None, second_embedding=None):
        super(HousePriceModel, self).__init__(idx_coor_map, config, n_epochs, embedding, second_embedding)

    def get_features(self, input_data):
        features = input_data[['numBedrooms', 'numBathrooms', 'sqft', 'region_coor', 'priceSqft', 'lat', 'lon']]

        embed = self.get_embedding()
        if embed is not None:

            embed_df = pd.DataFrame(embed, index=self.idx_coor_map.values())
            embed_features = pd.merge(features, embed_df, left_on='region_coor', right_index=True, how='inner')
            embed_features.drop('region_coor', axis=1, inplace=True)

            X = embed_features.drop(['priceSqft', 'lat', 'lon'], axis=1).values
            y = embed_features['priceSqft'].values

        else:
            embed = load_embedding(self.config['embedding_file'])
            embed_df = pd.DataFrame(embed, index=self.idx_coor_map.values())
            embed_features = pd.merge(features, embed_df, left_on='region_coor', right_index=True, how='inner')
            X = embed_features[['numBedrooms', 'numBathrooms', 'sqft']].values
            y = embed_features['priceSqft'].values

        print(X.shape)
        self.X = X
        self.y = y

class TrafficVolumeModel(PredictionModel):
    def __init__(self, idx_coor_map, config, n_epochs, embedding_fname=None, second_embed_fname=None):
        super(TrafficVolumeModel, self).__init__(idx_coor_map, config, n_epochs, embedding_fname, second_embed_fname)

    def get_features(self, input_data):
        features = input_data[['region_coor', 'Latitude', 'Longitude', 'Total Passing Vehicle Volume']]

        embed = self.get_embedding()
        if embed is not None:

            embed_df = pd.DataFrame(embed, index=self.idx_coor_map.values())
            embed_features = pd.merge(features, embed_df, left_on='region_coor', right_index=True, how='inner')
            embed_features.drop('region_coor', axis=1, inplace=True)

            X = embed_features.drop(['Latitude', 'Longitude', 'Total Passing Vehicle Volume'], axis=1).values
            y = embed_features['Total Passing Vehicle Volume'].values

        else:
            embed = load_embedding(self.config['embedding_file'])
            embed_df = pd.DataFrame(embed, index=self.idx_coor_map.values())
            embed_features = pd.merge(features, embed_df, left_on='region_coor', right_index=True, how='inner')
            X = embed_features[['Latitude', 'Longitude', 'Total Passing Vehicle Volume']].values
            y = embed_features['Total Passing Vehicle Volume'].values

        print(X.shape)
        self.X = X
        self.y = y