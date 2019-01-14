import sys
import os
import pandas as pd
import numpy as np
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config import get_config
from grid.create_grid import RegionGrid
import xgboost
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import KFold
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

    def get_features(self, zillow):
        features = zillow[['numBedrooms', 'numBathrooms', 'sqft', 'region_coor', 'priceSqft', 'lat', 'lon']]

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

    def train_eval(self, train_idx, test_idx):

        trn = xgboost.DMatrix(self.X[train_idx, :], label=self.y[train_idx])
        tst = xgboost.DMatrix(self.X[test_idx, :], label=self.y[test_idx])

        eval_list = [(trn, 'train')]
        model = xgboost.train(self.param, trn, self.n_epochs, verbose_eval=True, evals=eval_list)
        pred = model.predict(tst)
        
        rmse = np.sqrt(mean_squared_error(self.y[test_idx], pred))
        mae = mean_absolute_error(self.y[test_idx], pred)

        return rmse, mae

if len(sys.argv) > 1:
    n_epochs = int(sys.argv[1])
else:
    n_epochs = 150

n_folds = 5

print("K-Fold Prediction - training epochs: {}".format(n_epochs))



c = get_config()
region_grid = RegionGrid(config=c)
region_grid.load_weighted_mtx()
zillow = region_grid.load_housing_data(c['housing_data_file'])



# init naive
naive_mod = re_mod = PredictionModel(region_grid.idx_coor_map, c, n_epochs)
naive_mod.get_features(zillow)

# init deepwalk
deepwalk_mod = PredictionModel(region_grid.idx_coor_map, c, n_epochs, c['deepwalk_file'])
deepwalk_mod.get_features(zillow)

# init nmf
nmf_mod = PredictionModel(region_grid.idx_coor_map, c, n_epochs, c['nmf_file'])
nmf_mod.get_features(zillow)

# init region encoder
re_mod = PredictionModel(region_grid.idx_coor_map, c, n_epochs, c['embedding_file'])
re_mod.get_features(zillow)

# init deepwalk + region encoder
joint_mod = PredictionModel(region_grid.idx_coor_map, c, n_epochs, c['embedding_file'], c['deepwalk_file'])
joint_mod.get_features(zillow)


k_fold = KFold(n_splits=n_folds, shuffle=True, random_state=1990)

naive_err = np.zeros((n_folds, 2))
deepwalk_err = np.zeros((n_folds, 2))
embed_err = np.zeros((n_folds, 2))
nmf_err = np.zeros((n_folds, 2))
deepwalk_and_proposed_err = np.zeros((n_folds, 2))


train_ind_arr = np.arange(deepwalk_mod.X.shape[0])

fold_cntr = 0
for train_idx, test_idx in k_fold.split(train_ind_arr):
    print("Beginning Fold: {}".format(fold_cntr+1))
    # Naive Model
    rmse, mae = naive_mod.train_eval(train_idx, test_idx)
    naive_err[fold_cntr, 0] = rmse
    naive_err[fold_cntr, 1] = mae


    # DeepWalk Model
    rmse, mae = deepwalk_mod.train_eval(train_idx, test_idx)
    deepwalk_err[fold_cntr, 0] = rmse
    deepwalk_err[fold_cntr, 1] = mae

    # Matrix Factorization Model
    rmse, mae = nmf_mod.train_eval(train_idx, test_idx)
    nmf_err[fold_cntr, 0] = rmse
    nmf_err[fold_cntr, 1] = mae


    # Embedding model
    rmse, mae = re_mod.train_eval(train_idx, test_idx)
    embed_err[fold_cntr, 0] = rmse
    embed_err[fold_cntr, 1] = mae

    # Joint Model
    rmse, mae = joint_mod.train_eval(train_idx, test_idx)
    deepwalk_and_proposed_err[fold_cntr, 0] = rmse
    deepwalk_and_proposed_err[fold_cntr, 1] = mae



    fold_cntr += 1

naive_err_mean = np.mean(naive_err, axis=0)
naive_err_std = np.std(naive_err, axis=0)

embed_err_mean = np.mean(embed_err, axis=0)
embed_err_std = np.std(embed_err, axis=0)

deepwalk_err_mean = np.mean(deepwalk_err, axis=0)
deepwalk_err_std = np.std(deepwalk_err, axis=0)

nmf_err_mean = np.mean(nmf_err, axis=0)
nmf_err_std = np.std(nmf_err, axis=0)

deepwalk_and_proposed_mean = np.mean(deepwalk_and_proposed_err, axis=0)
deepwalk_and_proposed_std = np.std(deepwalk_and_proposed_err, axis=0)


print("Naive Model:")
print('RMSE: {:.4f} ({:.4f})'.format(naive_err_mean[0], naive_err_std[0]))
print('MAE: {:.4f} ({:.4f})'.format(naive_err_mean[1], naive_err_std[1]))

print("Deepwalk Model:")
print('RMSE: {:.4f} ({:.4f})'.format(deepwalk_err_mean[0], deepwalk_err_std[0]))
print('MAE: {:.4f} ({:.4f})'.format(deepwalk_err_mean[1], deepwalk_err_std[1]))

print("Matrix Factorization Model:")
print('RMSE: {:.4f} ({:.4f})'.format(nmf_err_mean[0], nmf_err_std[0]))
print('MAE: {:.4f} ({:.4f})'.format(nmf_err_mean[1], nmf_err_std[1]))

print("Embedding Model:")
print('RMSE: {:.4f} ({:.4f})'.format(embed_err_mean[0], embed_err_std[0]))
print('MAE: {:.4f} ({:.4f})'.format(embed_err_mean[1], embed_err_std[1]))

print("Deepwalk + Proposed Model:")
print('RMSE: {:.4f} ({:.4f})'.format(deepwalk_and_proposed_mean[0], deepwalk_and_proposed_std[0]))
print('MAE: {:.4f} ({:.4f})'.format(deepwalk_and_proposed_mean[1], deepwalk_and_proposed_std[1]))

