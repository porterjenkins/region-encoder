import sys
import os
import pandas as pd
import numpy as np
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import xgboost
from sklearn.metrics import mean_squared_error, mean_absolute_error
from model.utils import load_embedding


class PredictionModel(object):
    def __init__(self, idx_coor_map, config, n_epochs, embedding_fname=None, second_embed_fname=None):
        self.idx_coor_map = idx_coor_map
        self.config = config
        self.n_epochs = n_epochs
        self.embedding_fname = embedding_fname
        self.second_embed_fname = second_embed_fname
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


    def get_features(self, input_data):
        pass


    def train_eval(self, train_idx, test_idx):

        trn = xgboost.DMatrix(self.X[train_idx, :], label=self.y[train_idx])
        tst = xgboost.DMatrix(self.X[test_idx, :], label=self.y[test_idx])

        eval_list = [(trn, 'train')]
        model = xgboost.train(self.param, trn, self.n_epochs, verbose_eval=True, evals=eval_list)
        pred = model.predict(tst)

        rmse = np.sqrt(mean_squared_error(self.y[test_idx], pred))
        mae = mean_absolute_error(self.y[test_idx], pred)

        return rmse, mae

class HousePriceModel(PredictionModel):
    def __init__(self, idx_coor_map, config, n_epochs, embedding_fname=None, second_embed_fname=None):
        super(HousePriceModel, self).__init__(idx_coor_map, config, n_epochs, embedding_fname, second_embed_fname)

    def get_features(self, input_data):
        features = input_data[['numBedrooms', 'numBathrooms', 'sqft', 'region_coor', 'priceSqft', 'lat', 'lon']]

        if self.embedding_fname is not None:
            embed = load_embedding(self.embedding_fname)

            if self.second_embed_fname is not None:
                second_embed = load_embedding(self.second_embed_fname)
                embed = np.concatenate((embed, second_embed), axis=1)

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

        if self.embedding_fname is not None:
            embed = load_embedding(self.embedding_fname)

            if self.second_embed_fname is not None:
                second_embed = load_embedding(self.second_embed_fname)
                embed = np.concatenate((embed, second_embed), axis=1)

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