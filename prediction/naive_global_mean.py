from numpy import loadtxt
import numpy as np
from sklearn.metrics import mean_squared_error

dataset = loadtxt('onehot_and_median_income.csv',
                  delimiter=",")
from sklearn.model_selection import train_test_split

# split data into X and y
X = dataset[:, 0:800]
Y = dataset[:, 801]

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=.2, random_state=1990)

y_train_mean = np.mean(y_train)
y_pred = np.full(shape=(len(y_test)), fill_value=y_train_mean)

test_err = np.sqrt(mean_squared_error(y_true=y_test, y_pred=y_pred))

print("Test RMSE: {}".format(test_err))
