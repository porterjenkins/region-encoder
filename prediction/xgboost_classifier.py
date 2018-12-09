import pandas
from sklearn import metrics
from xgboost import XGBClassifier

df = pandas.read_excel(
    "/home/ahmad/repos/region-representation-learning/chicago/Household Income by Race and Census Tract and Community Area.xlsx")

# drop the header rows
dropped = df.drop(df.index[:4])

ids = dropped['TRACT']
median = dropped['B1913001']
# create one hot
one_hot = pandas.get_dummies(ids, prefix="tract", drop_first=True)

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(one_hot, median, test_size=.2, random_state=0)

alg = XGBClassifier(learning_rate=0.1, n_estimators=140, max_depth=5, objective='binary:logistic', nthread=4)

print('Start Training')
alg.fit(X_train, y_train)

print("Start Predicting")

predictions = alg.predict(X_test)

print("F1 Score: %f" % metrics.f1_score(y_test, predictions))
print("Score: %.4g" % metrics.accuracy_score(y_test, predictions))
