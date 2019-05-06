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

elif task == 'check_in':
    input_data = region_grid.get_checkin_counts(metric="mean")
    # init prediction models
    re_mod = CheckinModel(region_grid.idx_coor_map, c, n_epochs, c['embedding_file'])
    autoencoder_mod = CheckinModel(region_grid.idx_coor_map, c, n_epochs, c['autoencoder_embedding_file'])


else:
    raise NotImplementedError("User must input task: {'house_price', or 'checkin'")


# get features
re_mod.get_features(input_data)
autoencoder_mod.get_features(input_data)


k_fold = KFold(n_splits=n_folds, shuffle=True, random_state=1990)


embed_err = np.zeros((n_folds, 2))
ae_err = np.zeros((n_folds, 2))

train_ind_arr = np.arange(re_mod.X.shape[0])


fold_cntr = 0
for train_idx, test_idx in k_fold.split(train_ind_arr):
    print("Beginning Fold: {}".format(fold_cntr+1))

    # RegionEncoder model
    rmse, mae = re_mod.train_eval(train_idx, test_idx, estimator)
    embed_err[fold_cntr, 0] = rmse
    embed_err[fold_cntr, 1] = mae

    #AutoEncoder Model

    rmse, mae = autoencoder_mod.train_eval(train_idx, test_idx, estimator)
    ae_err[fold_cntr, 0] = rmse
    ae_err[fold_cntr, 1] = mae


    fold_cntr += 1


results = []


ae_err_mean = np.mean(ae_err, axis=0)
ae_err_std = np.std(ae_err, axis=0)
results.append(['AutoEncoder', ae_err_mean[0], ae_err_std[0], ae_err_mean[1], ae_err_std[1]])


embed_err_mean = np.mean(embed_err, axis=0)
embed_err_std = np.std(embed_err, axis=0)
results.append(['RegionEncoder', embed_err_mean[0], embed_err_std[0], embed_err_mean[1], embed_err_std[1]])




results_df = pd.DataFrame(results, columns=['model', 'cv rmse', 'std rmse', 'cv mae', 'std mae'])
print(results_df)

results_df.to_csv("../results/ablation-{}-{}-results.csv".format(task, estimator))

# plot result



groups = np.arange(results_df.shape[0])
width = .35

fig, ax = plt.subplots(nrows=2, ncols=1)



ax[0].bar(groups, results_df['cv rmse'], width, label='rmse')
ax[1].bar(groups + width, results_df['cv mae'], width, label='mae')

plt.xticks(groups + width / 2, results_df['model'])
#ax.set_xticks(groups + width / 2, results_df['model'])

#plt.legend(loc='best')
plt.savefig("../results/ablation-{}-{}-results.pdf".format(task, estimator))