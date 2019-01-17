import sys
import os
import numpy as np
import pandas as pd
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config import get_config
from grid.create_grid import RegionGrid
from sklearn.model_selection import KFold
from experiments.prediction import HousePriceModel, TrafficVolumeModel



if len(sys.argv) == 1:
    raise Exception("User must input task, estimator")
else:
   task = sys.argv[1]
   estimator = sys.argv[2]

assert(estimator in ['xgb', 'logit', 'rf'])

n_epochs = 150
n_folds = 5

print("K-Fold Learning - {}".format(estimator))


c = get_config()
region_grid = RegionGrid(config=c)
region_grid.load_weighted_mtx()

tmp = pd.DataFrame(region_grid.feature_matrix, index = region_grid.idx_coor_map.values())

if task == 'house_price':
    input_data = region_grid.load_housing_data(c['housing_data_file'])
    # Initialize Models
    naive_mod = HousePriceModel(region_grid.idx_coor_map, c, n_epochs)
    naive_raw_feature_mod = HousePriceModel(region_grid.idx_coor_map, c, n_epochs, region_grid.feature_matrix,
                                            region_grid.weighted_mtx)
    deepwalk_mod = HousePriceModel(region_grid.idx_coor_map, c, n_epochs, c['deepwalk_file'])
    re_mod = HousePriceModel(region_grid.idx_coor_map, c, n_epochs, c['embedding_file'])
    joint_mod = HousePriceModel(region_grid.idx_coor_map, c, n_epochs, c['embedding_file'], c['deepwalk_file'])
    nmf_mod = HousePriceModel(region_grid.idx_coor_map, c, n_epochs, c['nmf_file'])
    pca_mod = HousePriceModel(region_grid.idx_coor_map, c, n_epochs, c['pca_file'])
    autoencoder_mod = HousePriceModel(region_grid.idx_coor_map, c, n_epochs, c['autoencoder_embedding_file'])
    joint_ae_dw = HousePriceModel(region_grid.idx_coor_map, c, n_epochs, c['deepwalk_file'],
                                  c['autoencoder_embedding_file'])

elif task == 'traffic':
    input_data = region_grid.load_traffic_data(c['traffic_data_file'])
    # Initialize Models
    naive_mod = TrafficVolumeModel(region_grid.idx_coor_map, c, n_epochs)
    naive_raw_feature_mod = TrafficVolumeModel(region_grid.idx_coor_map, c, n_epochs, region_grid.feature_matrix,
                                               region_grid.weighted_mtx)
    deepwalk_mod = TrafficVolumeModel(region_grid.idx_coor_map, c, n_epochs, c['deepwalk_file'])
    re_mod = TrafficVolumeModel(region_grid.idx_coor_map, c, n_epochs, c['embedding_file'])
    joint_mod = TrafficVolumeModel(region_grid.idx_coor_map, c, n_epochs, c['embedding_file'], c['deepwalk_file'])
    nmf_mod = TrafficVolumeModel(region_grid.idx_coor_map, c, n_epochs, c['nmf_file'])
    pca_mod = TrafficVolumeModel(region_grid.idx_coor_map, c, n_epochs, c['pca_file'])
    autoencoder_mod = TrafficVolumeModel(region_grid.idx_coor_map, c, n_epochs, c['autoencoder_embedding_file'])
    joint_ae_dw = HousePriceModel(region_grid.idx_coor_map, c, n_epochs, c['deepwalk_file'],
                                  c['autoencoder_embedding_file'])


else:
    raise NotImplementedError("User must input task: {'house_price' or 'traffic'")



# Get Features
naive_mod.get_features(input_data)
naive_raw_feature_mod.get_features(input_data)
deepwalk_mod.get_features(input_data)
nmf_mod.get_features(input_data)
re_mod.get_features(input_data)
joint_mod.get_features(input_data)
pca_mod.get_features(input_data)
autoencoder_mod.get_features(input_data)
joint_ae_dw.get_features(input_data)

k_fold = KFold(n_splits=n_folds, shuffle=True, random_state=1990)

naive_err = np.zeros((n_folds, 2))
raw_features_err = np.zeros((n_folds, 2))
deepwalk_err = np.zeros((n_folds, 2))
embed_err = np.zeros((n_folds, 2))
nmf_err = np.zeros((n_folds, 2))
pca_err = np.zeros((n_folds, 2))
ae_err = np.zeros((n_folds, 2))
ae_dw_err = np.zeros((n_folds, 2))


train_ind_arr = np.arange(deepwalk_mod.X.shape[0])

fold_cntr = 0
for train_idx, test_idx in k_fold.split(train_ind_arr):
    print("Beginning Fold: {}".format(fold_cntr+1))
    # Naive Model
    rmse, mae = naive_mod.train_eval(train_idx, test_idx, estimator)
    naive_err[fold_cntr, 0] = rmse
    naive_err[fold_cntr, 1] = mae

    # Naive model w/ raw features

    rmse, mae = naive_raw_feature_mod.train_eval(train_idx, test_idx, estimator)
    raw_features_err[fold_cntr, 0] = rmse
    raw_features_err[fold_cntr, 1] = mae

    # DeepWalk Model
    rmse, mae = deepwalk_mod.train_eval(train_idx, test_idx, estimator)
    deepwalk_err[fold_cntr, 0] = rmse
    deepwalk_err[fold_cntr, 1] = mae

    # Matrix Factorization Model
    rmse, mae = nmf_mod.train_eval(train_idx, test_idx, estimator)
    nmf_err[fold_cntr, 0] = rmse
    nmf_err[fold_cntr, 1] = mae


    # RegionEncoder model
    rmse, mae = re_mod.train_eval(train_idx, test_idx, estimator)
    embed_err[fold_cntr, 0] = rmse
    embed_err[fold_cntr, 1] = mae

    #PCA model
    rmse, mae = pca_mod.train_eval(train_idx, test_idx, estimator)
    pca_err[fold_cntr, 0] = rmse
    pca_err[fold_cntr, 1] = mae

    #AutoEncoder Model

    rmse, mae = autoencoder_mod.train_eval(train_idx, test_idx, estimator)
    ae_err[fold_cntr, 0] = rmse
    ae_err[fold_cntr, 1] = mae

    # AutoEncoder + DeepWalk
    rmse, mae = joint_ae_dw.train_eval(train_idx, test_idx, estimator)
    ae_dw_err[fold_cntr, 0] = rmse
    ae_dw_err[fold_cntr, 1] = mae



    fold_cntr += 1

results = []

naive_err_mean = np.mean(naive_err, axis=0)
naive_err_std = np.std(naive_err, axis=0)
results.append(['Naive', naive_err_mean[0], naive_err_std[0], naive_err_mean[1], naive_err_std[1]])

raw_features_mean = np.mean(raw_features_err, axis=0)
raw_features_std = np.std(raw_features_err, axis=0)
results.append(['Naive + raw features', raw_features_mean[0], raw_features_std[0], raw_features_mean[1], raw_features_std[1]])

deepwalk_err_mean = np.mean(deepwalk_err, axis=0)
deepwalk_err_std = np.std(deepwalk_err, axis=0)
results.append(['DeepWalk Embedding', deepwalk_err_mean[0], deepwalk_err_std[0], deepwalk_err_mean[1], deepwalk_err_std[1]])

nmf_err_mean = np.mean(nmf_err, axis=0)
nmf_err_std = np.std(nmf_err, axis=0)
results.append(['Matrix Factorization', nmf_err_mean[0], nmf_err_std[0], nmf_err_mean[1], nmf_err_std[1]])

pca_err_mean = np.mean(pca_err, axis=0)
pca_err_std = np.std(pca_err, axis=0)
results.append(['PCA', pca_err_mean[0], pca_err_std[0], pca_err_mean[1], pca_err_std[1]])

ae_err_mean = np.mean(ae_err, axis=0)
ae_err_std = np.std(ae_err, axis=0)
results.append(['AutoEncoder', ae_err_mean[0], ae_err_std[0], ae_err_mean[1], ae_err_std[1]])

joint_ae_dw_err_mean = np.mean(ae_dw_err, axis=0)
joint_ae_dw_err_std = np.std(ae_dw_err, axis=0)
results.append(['AutoEncoder + DeepWalk', joint_ae_dw_err_mean[0], joint_ae_dw_err_std[0], joint_ae_dw_err_mean[1], joint_ae_dw_err_std[1]])

embed_err_mean = np.mean(embed_err, axis=0)
embed_err_std = np.std(embed_err, axis=0)
results.append(['RegionEncoder', embed_err_mean[0], embed_err_std[0], embed_err_mean[1], embed_err_std[1]])



results_df = pd.DataFrame(results, columns=['model', 'cv mse', 'std mse', 'cv mae', 'std mae'])
print(results_df)