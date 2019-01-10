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

if len(sys.argv) > 1:
    n_epochs = int(sys.argv[1])
else:
    n_epochs = 250

n_folds = 5

print("K-Fold Prediction - training epochs: {}".format(n_epochs))


def get_train_test_idx(n, train_size=.8):
    idx = np.arange(n)
    idx = np.random.permutation(idx)

    train_k = int(train_size*n)
    test_k = n - train_k

    train_idx = idx[:train_k]
    test_idx = idx[-test_k:]

    return train_idx, test_idx



c = get_config()
region_grid = RegionGrid(config=c)
region_grid.load_weighted_mtx()
zillow = region_grid.load_housing_data(c['housing_data_file'])


param = {
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


features = zillow[['numBedrooms', 'numBathrooms', 'sqft', 'region_coor', 'priceSqft','lat', 'lon']]

re_embed = region_grid.load_embedding(c['embedding_file'])
re_df = pd.DataFrame(re_embed, index=region_grid.idx_coor_map.values())

deepwalk = region_grid.load_embedding(c['deepwalk_file'])
deepwalk_df = pd.DataFrame(deepwalk, index=region_grid.idx_coor_map.values())


re_features = pd.merge(features, re_df, left_on='region_coor', right_index=True, how='inner')
re_features.drop('region_coor', axis=1, inplace=True)
print(re_features.shape)

deepwalk_features = pd.merge(features, deepwalk_df, left_on='region_coor', right_index=True, how='inner')
deepwalk_features.drop('region_coor', axis=1, inplace=True)

print(deepwalk_features.shape)

k_fold = KFold(n_splits=n_folds, shuffle=True)

naive_err = np.zeros((n_folds, 2))
deepwalk_err = np.zeros((n_folds, 2))
embed_err = np.zeros((n_folds, 2))

train_ind_arr = np.arange(deepwalk_features.shape[0])

fold_cntr = 0
for train_idx, test_idx in k_fold.split(train_ind_arr):
    print("Beginning Fold: {}".format(fold_cntr+1))
    # Naive Model
    X = features[['numBedrooms', 'numBathrooms', 'sqft', 'lat', 'lon']].values
    y = features['priceSqft'].values


    trn = xgboost.DMatrix(X[train_idx, :], label=y[train_idx])
    tst = xgboost.DMatrix(X[test_idx, :], label=y[test_idx])


    eval_list = [(trn, 'train')]
    model = xgboost.train(param, trn, n_epochs, verbose_eval=True, evals=eval_list)
    pred = model.predict(tst)
    naive_err[fold_cntr, 0] = np.sqrt(mean_squared_error(y[test_idx], pred))
    naive_err[fold_cntr, 1] = mean_absolute_error(y[test_idx], pred)


    # DeepWalk Model

    X = deepwalk_features.drop('priceSqft', axis=1).values
    y = deepwalk_features['priceSqft'].values

    trn = xgboost.DMatrix(X[train_idx, :], label=y[train_idx])
    tst = xgboost.DMatrix(X[test_idx, :], label=y[test_idx])


    eval_list = [(trn, 'train')]
    model = xgboost.train(param, trn, n_epochs, verbose_eval=True, evals=eval_list)
    pred = model.predict(tst)
    rmse = np.sqrt(mean_squared_error(y[test_idx], pred))
    deepwalk_err[fold_cntr, 0] = np.sqrt(mean_squared_error(y[test_idx], pred))
    deepwalk_err[fold_cntr, 1] = mean_absolute_error(y[test_idx], pred)


    # Embedding model

    X = re_features.drop('priceSqft', axis=1).values
    y = re_features['priceSqft'].values

    trn = xgboost.DMatrix(X[train_idx, :], label=y[train_idx])
    tst = xgboost.DMatrix(X[test_idx, :], label=y[test_idx])


    eval_list = [(trn, 'train')]
    model = xgboost.train(param, trn, n_epochs, verbose_eval=True, evals=eval_list)
    pred = model.predict(tst)
    rmse = np.sqrt(mean_squared_error(y[test_idx], pred))
    embed_err[fold_cntr, 0] = np.sqrt(mean_squared_error(y[test_idx], pred))
    embed_err[fold_cntr, 1] = mean_absolute_error(y[test_idx], pred)

    fold_cntr += 1

naive_err_mean = np.mean(naive_err, axis=0)
naive_err_std = np.std(naive_err, axis=0)

embed_err_mean = np.mean(embed_err, axis=0)
embed_err_std = np.std(embed_err, axis=0)

deepwalk_err_mean = np.mean(deepwalk_err, axis=0)
deepwalk_err_std = np.std(deepwalk_err, axis=0)


print("Naive Model:")
print('RMSE: {:.4f} ({:.4f})'.format(naive_err_mean[0], naive_err_std[0]))
print('MAE: {:.4f} ({:.4f})'.format(naive_err_mean[1], naive_err_std[1]))

print("Deepwalk Model:")
print('RMSE: {:.4f} ({:.4f})'.format(deepwalk_err_mean[0], deepwalk_err_std[0]))
print('MAE: {:.4f} ({:.4f})'.format(deepwalk_err_mean[1], deepwalk_err_std[1]))


print("Embedding Model:")
print('RMSE: {:.4f} ({:.4f})'.format(embed_err_mean[0], embed_err_std[0]))
print('MAE: {:.4f} ({:.4f})'.format(embed_err_mean[1], embed_err_std[1]))
