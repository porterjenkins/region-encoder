import numpy as np
import xgboost
from numpy import loadtxt
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split
import pandas as pd
from config import get_config
import pandas as pd



c = get_config()
dataset = pd.read_csv('onehot_and_median_income.csv',
                  delimiter=",",header=None, index_col=0)

deepwalk_features = list()
idx = list()

with open(c['deepwalk_file'], 'rb') as f:
    cntr = 0
    for line in f:
        if cntr > 0:
            row = line.decode('utf-8').split(" ")
            row_float = []
            for i, element in enumerate(row):
                # skip 0th element - tract id
                if i == 0:
                    idx.append(int(element))
                else:
                    row_float.append(float(element))
            deepwalk_features.append(row_float)

        cntr +=1

deepwalk_features = pd.DataFrame(deepwalk_features, index=idx)
# split data into X and y
#X = dataset[:, 0:800]
Y = dataset.iloc[:, 801].to_frame()


X = pd.merge(deepwalk_features, Y, left_index=True, right_index=True, how='left')


X_all = X.iloc[:, 0:64].values
Y_all = X.iloc[:, 64].values

X_train, X_test, y_train, y_test = train_test_split(X_all, Y_all, test_size=.2, random_state=1990)
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
trn = xgboost.DMatrix(X_train, label=y_train)
tst = xgboost.DMatrix(X_test, label=y_test)
res = xgboost.cv(param, trn, nfold=4, early_stopping_rounds=50, metrics=['rmse'], maximize=False)
min_index = np.argmin(res['test-rmse-mean'])

model = xgboost.train(param, trn, min_index, [(trn, 'train'), (tst, 'test')])
pred = model.predict(tst)
rmse = np.sqrt(mean_squared_error(y_test, pred))
mae = mean_absolute_error(y_test, pred)
print('Test RMSE: {:.4f}'.format(rmse))
print('Test MAE: {:.4f}'.format(mae))