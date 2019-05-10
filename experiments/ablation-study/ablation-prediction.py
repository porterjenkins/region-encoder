import sys
import os
import numpy as np
import pandas as pd
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from config import get_config
from grid.create_grid import RegionGrid
from sklearn.model_selection import KFold
from experiments.prediction import HousePriceModel, TrafficVolumeModel, CheckinModel
import matplotlib.pyplot as plt

RANDOM_STATE = 1990

if len(sys.argv) == 1:
    raise Exception("User must input task, estimator")
else:
    task = sys.argv[1]
    estimator = sys.argv[2]
    try:
       n_epochs = int(sys.argv[3])
    except IndexError:
        n_epochs = 25


assert(estimator in ['xgb', 'lasso', 'rf', 'mlp', 'ridge'])
n_folds = 5
print("K-Fold Learning - {}".format(estimator))


c = get_config()
region_grid = RegionGrid(config=c)
region_grid.load_weighted_mtx()

tmp = pd.DataFrame(region_grid.feature_matrix, index = region_grid.idx_coor_map.values())

autoencoder_embed = c['autoencoder_embedding_file']
regionencoder_embed = c['embedding_file']
gcn_all_embed = '{}gcn_all_embedding.txt'.format(c['data_dir_main'])
gcn_sg_embed = '{}gcn_skipgram_embedding.txt'.format(c['data_dir_main'])
gcn_flow_embed = '{}gcn_flow_embedding.txt'.format(c['data_dir_main'])
concat_embed ='{}concat_global_embedding.txt'.format(c['data_dir_main'])



if task == 'house_price':
    input_data = region_grid.load_housing_data(c['housing_data_file'])
    re_mod = HousePriceModel(region_grid.idx_coor_map, c, n_epochs, c['embedding_file'])
    autoencoder_mod = HousePriceModel(region_grid.idx_coor_map, c, n_epochs, c['autoencoder_embedding_file'])
    gcn_skipgram_mod = HousePriceModel(region_grid.idx_coor_map, c, n_epochs, gcn_sg_embed)
    gcn_flow_mod = HousePriceModel(region_grid.idx_coor_map, c, n_epochs, gcn_flow_embed)
    gcn_all_mod = HousePriceModel(region_grid.idx_coor_map, c, n_epochs, gcn_all_embed)
    gcn_ae_concat_mod = HousePriceModel(region_grid.idx_coor_map, c, n_epochs, c['autoencoder_embedding_file'],
                                     gcn_all_embed)


elif task == 'check_in':
    input_data = region_grid.get_checkin_counts(metric="mean")
    # init prediction models
    re_mod = CheckinModel(region_grid.idx_coor_map, c, n_epochs, c['embedding_file'])
    gcn_all_mod = CheckinModel(region_grid.idx_coor_map, c, n_epochs, gcn_all_embed)
    gcn_skipgram_mod = CheckinModel(region_grid.idx_coor_map, c, n_epochs, gcn_sg_embed)
    gcn_flow_mod = CheckinModel(region_grid.idx_coor_map, c, n_epochs, gcn_flow_embed)
    gcn_ae_concat_mod = CheckinModel(region_grid.idx_coor_map, c, n_epochs, c['autoencoder_embedding_file'],
                                     gcn_all_embed)
    autoencoder_mod = CheckinModel(region_grid.idx_coor_map, c, n_epochs, c['autoencoder_embedding_file'])


else:
    raise NotImplementedError("User must input task: {'house_price', or 'check_in}'")


# get features
re_mod.get_features(input_data)
autoencoder_mod.get_features(input_data)
gcn_all_mod.get_features(input_data)
gcn_skipgram_mod.get_features(input_data)
gcn_flow_mod.get_features(input_data)
gcn_ae_concat_mod.get_features(input_data)



k_fold = KFold(n_splits=n_folds, shuffle=True, random_state=1990)


embed_err = np.zeros((n_folds, 2))
ae_err = np.zeros((n_folds, 2))
gcn_all_err = np.zeros((n_folds, 2))
gcn_flow_err = np.zeros((n_folds, 2))
gcn_sg_err = np.zeros((n_folds, 2))
gcn_ae_concat_err = np.zeros((n_folds, 2))


train_ind_arr = np.arange(re_mod.X.shape[0])


fold_cntr = 0
for train_idx, test_idx in k_fold.split(train_ind_arr):
    print("Beginning Fold: {}".format(fold_cntr+1))

    # RegionEncoder model
    rmse, mae = re_mod.train_eval(train_idx, test_idx, estimator, random_state=RANDOM_STATE)
    embed_err[fold_cntr, 0] = rmse
    embed_err[fold_cntr, 1] = mae

    #AutoEncoder Model
    rmse, mae = autoencoder_mod.train_eval(train_idx, test_idx, estimator, random_state=RANDOM_STATE)
    ae_err[fold_cntr, 0] = rmse
    ae_err[fold_cntr, 1] = mae

    # GCN all
    rmse, mae = gcn_all_mod.train_eval(train_idx, test_idx, estimator, random_state=RANDOM_STATE)
    gcn_all_err[fold_cntr, 0] = rmse
    gcn_all_err[fold_cntr, 1] = mae

    # GCN flow
    rmse, mae = gcn_flow_mod.train_eval(train_idx, test_idx, estimator, random_state=RANDOM_STATE)
    gcn_flow_err[fold_cntr, 0] = rmse
    gcn_flow_err[fold_cntr, 1] = mae

    # GCN skipgram
    rmse, mae = gcn_skipgram_mod.train_eval(train_idx, test_idx, estimator, random_state=RANDOM_STATE)
    gcn_sg_err[fold_cntr, 0] = rmse
    gcn_sg_err[fold_cntr, 1] = mae

    # GCN + AE (concat)
    rmse, mae = gcn_ae_concat_mod.train_eval(train_idx, test_idx, estimator, random_state=RANDOM_STATE)
    gcn_ae_concat_err[fold_cntr, 0] = rmse
    gcn_ae_concat_err[fold_cntr, 1] = mae


    fold_cntr += 1


results = []


ae_err_mean = np.mean(ae_err, axis=0)
ae_err_std = np.std(ae_err, axis=0)
results.append(['AutoEncoder', ae_err_mean[0], ae_err_std[0], ae_err_mean[1], ae_err_std[1]])

gcn_sg_mean = np.mean(gcn_sg_err, axis=0)
gcn_sg_std = np.std(gcn_sg_err, axis=0)
results.append(['GCN-SG', gcn_sg_mean[0], gcn_sg_std[0], gcn_sg_mean[1], gcn_sg_std[1]])

gcn_flow_mean = np.mean(gcn_flow_err, axis=0)
gcn_flow_std = np.std(gcn_flow_err, axis=0)
results.append(['GCN-flow', gcn_flow_mean[0], gcn_flow_std[0], gcn_flow_mean[1], gcn_flow_std[1]])

gcn_all_mean = np.mean(gcn_all_err, axis=0)
gcn_all_std = np.std(gcn_all_err, axis=0)
results.append(['GCN-SG-flow', gcn_all_mean[0], gcn_all_std[0], gcn_all_mean[1], gcn_all_std[1]])

concat_err_mean = np.mean(gcn_ae_concat_err, axis=0)
concat_err_std = np.std(embed_err, axis=0)
results.append(['GCN-SG-flow + AE', concat_err_mean[0], concat_err_std[0], concat_err_mean[1], concat_err_std[1]])


embed_err_mean = np.mean(embed_err, axis=0)
embed_err_std = np.std(embed_err, axis=0)
results.append(['RegionEncoder', embed_err_mean[0], embed_err_std[0], embed_err_mean[1], embed_err_std[1]])





results_df = pd.DataFrame(results, columns=['model', 'cv rmse', 'std rmse', 'cv mae', 'std mae'])
print(results_df)

results_df.to_csv("../results/ablation-{}-{}-{}-results.csv".format(c['city_name'], task, estimator))


# plot result
y_min_rmse = results_df['cv rmse'].min()*.95
y_max_rmse = results_df['cv rmse'].max()

y_min_mae = results_df['cv mae'].min()*.95
y_max_mae = results_df['cv mae'].max()

results_df['cv rmse'].plot(kind='bar', figsize=(8,8), fontsize=20, label='RMSE', alpha=.75)
plt.subplots_adjust(bottom=.27)
#plt.subplots_adjust(left=.19)
plt.xticks(range(results_df.shape[0]), results_df['model'], rotation=45)
plt.ylim((y_min_rmse, y_max_rmse))
plt.legend(loc='best', fontsize=24)
#plt.ylabel("MAE", fontsize=28)

plt.savefig("../results/ablation-{}-{}-{}-rmse.pdf".format(c['city_name'], task, estimator))
plt.clf()

results_df['cv mae'].plot(kind='bar', figsize=(8,8), fontsize=20, color='orange', alpha=.75, label='MAE')
plt.subplots_adjust(bottom=.27)
#plt.subplots_adjust(left=.17)
plt.xticks(range(results_df.shape[0]), results_df['model'], rotation=45)
plt.ylim((y_min_mae, y_max_mae))
plt.legend(loc='best', fontsize=24)
#plt.ylabel("MAE", fontsize=28)

plt.savefig("../results/ablation-{}-{}-{}-mae.pdf".format(c['city_name'], task, estimator))