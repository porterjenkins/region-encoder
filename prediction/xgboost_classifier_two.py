import numpy
import xgboost
from numpy import loadtxt
from sklearn.metrics import mean_squared_error, mean_absolute_error

n_epochs = 250
dataset = loadtxt('onehot_and_median_income.csv',
                  delimiter=",")
from sklearn.model_selection import train_test_split


# split data into X and y
X = dataset[:, 1:801]
Y = dataset[:, 802]

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
