import pandas
from sklearn.preprocessing import OneHotEncoder

df = pandas.read_excel(
    "/home/ahmad/repos/region-representation-learning/chicago/Household Income by Race and Census Tract and Community Area.xlsx")

# drop the header rows
df.drop(df.index[:2], inplace=True)
data = df[['TRACT', 'B1913001']]
# drop na rows
data = data.dropna(subset=['TRACT'])
# create one hots
ids = data['TRACT'].values
ids = ids.reshape(data.shape[0], 1)
one_hot_encoder = OneHotEncoder(sparse=False)
encoded = one_hot_encoder.fit_transform(ids)

median = data['B1913001'].values

new_df = pandas.DataFrame(encoded)
new_df['price'] = median

new_df.to_csv('onehot_and_median_income.csv', sep=",", index_label=False, index=False)
