import pandas as pd
import matplotlib.pyplot as plt
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from config import get_config


if len(sys.argv) == 1:
    raise Exception("User must input task, estimator")
else:
    task = sys.argv[1]
    estimator = sys.argv[2]
    try:
       n_epochs = int(sys.argv[3])
    except IndexError:
        n_epochs = 25

c = get_config()



results_df = pd.read_csv("../results/ablation-{}-{}-{}-results.csv".format(c['city_name'], task, estimator))


# plot result
y_min_rmse = results_df['cv rmse'].min()*.95
y_max_rmse = results_df['cv rmse'].max()*1.05

y_min_mae = results_df['cv mae'].min()*.95
y_max_mae = results_df['cv mae'].max()*1.05

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
