from numpy import loadtxt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier

dataset = loadtxt('/home/ahmad/repos/region-representation-learning/prediction/onehot_and_median_income.csv',
                  delimiter=",")
from sklearn.model_selection import train_test_split

# split data into X and y
X = dataset[:, 0:800]
Y = dataset[:, 801]

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=.2, random_state=0)

model = XGBClassifier(learning_rate=0.1, n_estimators=140, max_depth=5, objective='binary:logistic', nthread=4)

print('Start Training')
model.fit(X_train, y_train)
regression = LinearRegression()
regression.fit(X_train, y_train)

print("Start Predicting")

predictions = model.predict(X_test)
lin_pred = regression.predict(X_test)


print(regression.score(X_test, y_test))
print(model.score(X_test, y_test))

