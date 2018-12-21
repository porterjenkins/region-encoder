import pandas
from sklearn.preprocessing import OneHotEncoder
from config import get_config
import numpy as np

c = get_config()
df = pandas.read_excel(c['census_income_file'], dtype={'TRACT': str})

# drop the header rows
df.drop(df.index[:2], inplace=True)
data = df[['TRACT', 'B1913001']]
data.index = "17031" + data['TRACT']
# drop na rows
data['TRACT'] = data['TRACT'].astype(np.float32)
data = data.dropna(subset=['TRACT'])
# create one hots
ids = data['TRACT'].values
ids = ids.reshape(data.shape[0], 1)
one_hot_encoder = OneHotEncoder(sparse=False)
encoded = one_hot_encoder.fit_transform(ids)

median = data['B1913001'].values

new_df = pandas.DataFrame(encoded, index=data.index)
new_df['price'] = median

new_df.to_csv('onehot_and_median_income.csv', sep=",", header=None)
