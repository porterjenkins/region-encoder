import sys
import os
import pandas as pd
import numpy as np
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import xgboost
from sklearn.metrics import mean_squared_error, mean_absolute_error
from model.utils import load_embedding
from sklearn.linear_model import Lasso, Ridge
from sklearn.ensemble import RandomForestRegressor
from experiments.mlp import MLP
import torch
from sklearn.neural_network import MLPRegressor
from experiments.metrics import *
from sklearn.preprocessing import normalize, scale

class PredictionModel(object):
    def __init__(self, idx_coor_map, config, n_epochs, embedding=None, second_embedding=None, third_embedding=None):
        self.idx_coor_map = idx_coor_map
        self.config = config
        self.n_epochs = n_epochs
        self.embedding = embedding
        self.second_embedding = second_embedding
        self.third_embedding = third_embedding
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
                if self.third_embedding is not None:
                    third_embed = self.__is_fname(self.third_embedding)
                    embed = np.concatenate((embed, third_embed), axis=1)

            return embed
        else:
            return None


    def get_features(self, input_data):
        pass


    def train_eval(self, train_idx, test_idx, model='xgb'):

        # TODO: Fix random seed for all regressors?
        if model == 'xgb':
            trn = xgboost.DMatrix(self.X[train_idx, :], label=self.y[train_idx])
            tst = xgboost.DMatrix(self.X[test_idx, :], label=self.y[test_idx])

            eval_list = [(trn, 'train')]
            model = xgboost.train(self.param, trn, self.n_epochs, verbose_eval=True, evals=eval_list)
            pred = model.predict(tst)
        elif model == 'lasso':

            model = Lasso(alpha=.5, fit_intercept=True)
            model.fit(X=self.X[train_idx, :], y=self.y[train_idx])
            pred = model.predict(X=self.X[test_idx])

        elif model == 'ridge':

            model = Ridge(alpha=1.0, fit_intercept=True)
            model.fit(X=self.X[train_idx, :], y=self.y[train_idx])
            pred = model.predict(X=self.X[test_idx])

        elif model == 'rf':

            model = RandomForestRegressor()
            model.fit(X=self.X[train_idx, :], y=self.y[train_idx])
            pred = model.predict(X=self.X[test_idx])

        elif model == 'mlp':

            model = MLPRegressor(hidden_layer_sizes=(128,), activation='relu', solver='adam', alpha=.01, batch_size=100,
                                 learning_rate_init=.05, max_iter=self.n_epochs, random_state=1990)
            model.fit(X=self.X[train_idx, :], y=self.y[train_idx])
            pred = model.predict(X=self.X[test_idx])


        rmse = np.sqrt(mean_squared_error(self.y[test_idx], pred))
        mae = mean_absolute_error(self.y[test_idx], pred)
        #mre = mean_relative_error(self.y[test_idx], pred)

        return rmse, mae

class CheckinModel(PredictionModel):
    def __init__(self, idx_coor_map, config, n_epochs, embedding=None, second_embedding=None, third_embedding=None):
        super(CheckinModel, self).__init__(idx_coor_map, config, n_epochs, embedding, second_embedding, third_embedding)

    def get_features(self, input_data, norm_y=False):

        X = input_data[['lat', 'lon']]
        embed = self.get_embedding()
        if embed is not None:
            #self.X = np.concatenate((X.values, embed), axis=1)
            self.X = embed

        else:
            self.X = X.values

        if norm_y:
            self.y = scale(input_data['checkins'].values.reshape(-1,1), with_std=True, with_mean=True)
            #self.y = np.log(input_data['checkins'].values)
        else:
            self.y = input_data['checkins'].values

        print(self.X.shape)


class HousePriceModel(PredictionModel):
    def __init__(self, idx_coor_map, config, n_epochs, embedding=None, second_embedding=None, third_embedding=None):
        super(HousePriceModel, self).__init__(idx_coor_map, config, n_epochs, embedding, second_embedding, third_embedding)

    def get_features(self, input_data):
        features = input_data[['numBedrooms', 'numBathrooms', 'sqft', 'region_coor', 'priceSqft', 'lat', 'lon']]

        embed = self.get_embedding()
        if embed is not None:

            embed_df = pd.DataFrame(embed, index=self.idx_coor_map.values())
            embed_features = pd.merge(features, embed_df, left_on='region_coor', right_index=True, how='inner')
            embed_features.drop('region_coor', axis=1, inplace=True)

            X = embed_features.drop(['priceSqft','lat','lon'], axis=1).values
            y = embed_features['priceSqft'].values

        else:
            embed = load_embedding(self.config['embedding_file'])
            embed_df = pd.DataFrame(embed, index=self.idx_coor_map.values())
            embed_features = pd.merge(features, embed_df, left_on='region_coor', right_index=True, how='inner')
            embed_features = pd.get_dummies(embed_features, columns=['region_coor'])
            keepcols = [str(c) for c in list(embed_features.columns) if 'region_coor' in str(c)] + ['numBedrooms', 'numBathrooms', 'sqft']

            X = embed_features[keepcols].values
            y = embed_features['priceSqft'].values

        print(X.shape)
        self.X = X
        self.y = y

class TrafficVolumeModel(PredictionModel):
    def __init__(self, idx_coor_map, config, n_epochs, embedding_fname=None, second_embed_fname=None, third_embedding=None):
        super(TrafficVolumeModel, self).__init__(idx_coor_map, config, n_epochs, embedding_fname, second_embed_fname, third_embedding)

    def get_features(self, input_data):
        input_data = input_data[~np.isnan(input_data.traffic)]
        #features = input_data[['region_coor', 'hour', 'Direction', 'SHAPE_Leng','traffic']]
        features = input_data[['region_coor', 'Direction', 'SHAPE_Leng', 'traffic']]
        #features = pd.get_dummies(features, columns=['hour', 'Direction'])
        features = pd.get_dummies(features, columns=['Direction'])

        embed = self.get_embedding()
        if embed is not None:

            embed_df = pd.DataFrame(embed, index=self.idx_coor_map.values())
            embed_features = pd.merge(features, embed_df, left_on='region_coor', right_index=True, how='inner')
            embed_features.drop('region_coor', axis=1, inplace=True)



        else:
            embed = load_embedding(self.config['embedding_file'])
            embed_df = pd.DataFrame(embed, index=self.idx_coor_map.values())
            embed_features = pd.merge(features, embed_df, left_on='region_coor', right_index=True, how='inner')
            h_dim = int(self.config['hidden_dim_size'])
            drop_cols = list(range(h_dim)) + ['region_coor']
            embed_features.drop(drop_cols, axis=1, inplace=True)

        X = embed_features.drop(['traffic'], axis=1).values
        y = embed_features['traffic'].values


        print(X.shape)
        self.X = X
        self.y = y