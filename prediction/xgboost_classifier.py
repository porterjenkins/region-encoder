import pandas
from sklearn import metrics
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBClassifier

df = pandas.read_excel(
    "/home/ahmad/repos/region-representation-learning/chicago/Household Income by Race and Census Tract and Community Area.xlsx")

# drop the header rows
df.drop(df.index[:2], inplace=True)

data = df[['TRACT', 'B1913001']]

# data = data[pandas.notnull(df['TRACT'])]
data = data.dropna(subset=['TRACT'])
# create one hots
ids = data['TRACT'].values
ids = ids.reshape(data.shape[0], 1)
one_hot_encoder = OneHotEncoder(sparse=False)
encoded = one_hot_encoder.fit_transform(ids)


print(encoded)

median = data['B1913001']
# one_hot = pandas.get_dummies(ids, , drop_first=True)
#
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(encoded, median, test_size=.2, random_state=0)

alg = XGBClassifier(learning_rate=0.1, n_estimators=140, max_depth=5, objective='binary:logistic', nthread=4)

print('Start Training')
alg.fit(X_train, y_train)

print("Start Predicting")

predictions = alg.predict(X_test)

print("F1 Score: %f" % metrics.f1_score(y_test, predictions))
print("Score: %.4g" % metrics.accuracy_score(y_test, predictions))
