import numpy
import xgboost
from numpy import loadtxt
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import MultiLabelBinarizer
from config import get_config
import pandas as pd
import numpy as np

n_epochs = 250
dataset = pd.read_csv('onehot_and_median_income.csv',
                  delimiter=",", header=None, index_col=0)
from sklearn.model_selection import train_test_split


# split data into X and y
Y = dataset.iloc[:, 801].to_frame()

c = get_config()
idx = list()
adj_features = list()
# get adjacency list
with open(c['edge_list_file'], 'rb') as f:
    cntr = 0
    for line in f:
        row = line.decode('utf-8').split(" ")
        row_float = []
        for i, element in enumerate(row):
            # skip 0th element - tract id
            if i == 0:
                idx.append(int(element))
            else:
                if element != '\n':
                    row_float.append(element)
        adj_features.append(row_float)

        cntr +=1


encoder = MultiLabelBinarizer()
adj_features = encoder.fit_transform(adj_features)
tmp = np.where(adj_features[1, :] > 0)[0]
adj_features = pd.DataFrame(adj_features, index=idx)

classes = encoder.classes_

print("Test case: {}".format(idx[1]))
for i in tmp:
    print("--> {}".format(classes[i]))



X = pd.merge(adj_features, Y, left_index=True, right_index=True, how='left')
X_all = X.iloc[:, 0:801].values
Y_all = X.iloc[:, 801].values

X_train, X_test, y_train, y_test = train_test_split(X_all, Y_all, test_size=.2, random_state=1990)


X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=.2, random_state=1990)
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
#res = xgboost.cv(param, trn, nfold=4, early_stopping_rounds=50,metrics=['rmse'], maximize=False)
#min_index = numpy.argmin(res['test-rmse-mean'])

eval_list = [(trn, 'train'), (tst, 'test')]
model = xgboost.train(param, trn, n_epochs, verbose_eval=True, evals=eval_list)
#model = xgboost.train(param, trn, min_index, [(trn, 'train'), (tst, 'test')])
pred = model.predict(tst)
rmse = numpy.sqrt(mean_squared_error(y_test, pred))
mae = mean_absolute_error(y_test, pred)
print('Test RMSE: {:.4f}'.format(rmse))
print('Test MAE: {:.4f}'.format(mae))
