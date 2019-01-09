import sys
import os
import pandas as pd
import numpy as np
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config import get_config
from grid.create_grid import RegionGrid
import xgboost
from sklearn.metrics import mean_squared_error, mean_absolute_error



def get_train_test_idx(n, train_size=.8):
    idx = np.arange(n)
    idx = np.random.permutation(idx)

    train_k = int(train_size*n)
    test_k = n - train_k

    train_idx = idx[:train_k]
    test_idx = idx[-test_k:]

    return train_idx, test_idx



c = get_config()
region_grid = RegionGrid(config=c, load_imgs=False)
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

n_epochs = 100


features = zillow[['numBedrooms', 'numBathrooms', 'sqft', 'region_coor', 'priceSqft']]

re_embed = region_grid.load_embedding(c['embedding_file'])
re_df = pd.DataFrame(re_embed, index=region_grid.idx_coor_map.values())

features = pd.merge(features, re_df, left_on='region_coor', right_index=True, how='inner')
features.drop('region_coor', axis=1, inplace=True)



# Naive Model
X = features[['numBedrooms', 'numBathrooms', 'sqft']].values
y = features['priceSqft'].values


n = X.shape[0]
train_idx, test_idx = get_train_test_idx(n, .8)


trn = xgboost.DMatrix(X[train_idx, :], label=y[train_idx])
tst = xgboost.DMatrix(X[test_idx, :], label=y[test_idx])


eval_list = [(trn, 'train')]
model = xgboost.train(param, trn, n_epochs, verbose_eval=True, evals=eval_list)
pred = model.predict(tst)
rmse = np.sqrt(mean_squared_error(y[test_idx], pred))
mae = mean_absolute_error(y[test_idx], pred)

print("Naive Model: ")
print('Test RMSE: {:.4f}'.format(rmse))
print('Test MAE: {:.4f}'.format(mae))



# Embedding model

X = features.drop('priceSqft', axis=1).values
y = features['priceSqft'].values

trn = xgboost.DMatrix(X[train_idx, :], label=y[train_idx])
tst = xgboost.DMatrix(X[test_idx, :], label=y[test_idx])


eval_list = [(trn, 'train')]
model = xgboost.train(param, trn, n_epochs, verbose_eval=True, evals=eval_list)
pred = model.predict(tst)
rmse = np.sqrt(mean_squared_error(y[test_idx], pred))
mae = mean_absolute_error(y[test_idx], pred)

print("Embedding Model: ")
print('Test RMSE: {:.4f}'.format(rmse))
print('Test MAE: {:.4f}'.format(mae))

