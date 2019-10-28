import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config import get_config
from grid.create_grid import RegionGrid
import numpy as np
import matplotlib.pyplot as plt
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
import geopandas as gpd



c = get_config()





region_grid = RegionGrid(config=c)


house_price = region_grid.load_housing_data(c['housing_data_file'])
print(house_price.head())
checkin = region_grid.get_checkin_counts(metric="mean")

trim = np.percentile(house_price['priceSqft'], 99)
print(trim)

house_price = house_price[house_price['priceSqft'] < trim]
house_price.to_csv(c['data_dir_main'] + 'zillow_house_price_trim.csv', index=False)
checkin.to_csv(c['data_dir_main'] + "checkin.csv", index=False)
print(checkin.head())

#checkin.checkins.hist()
#plt.show()
#plt.clf()


#house_price['priceSqft'].hist(bins=50)
#plt.savefig('results/hist-{}-{}.pdf'.format('house-price', c['city_name']))




if c["city_name"] == 'nyc':
    scaler = 50
    scatter_size = 20
else:
    scaler = 10
    scatter_size = 4

city = gpd.read_file(c['shape_file'])
city.plot(alpha=.5, figsize=(10, 10), color='gray')

x = house_price.lon
y = house_price.lat
z = house_price['priceSqft']

cm = plt.cm.get_cmap('RdYlGn')
sc = plt.scatter(x, y, c=z, cmap=cm, alpha=.4,s=scatter_size)

#plt.ylim((c['lat_min'], c['lat_max']))
#plt.xlim((c['lon_min'], c['lon_max']))
cbar = plt.colorbar(sc)
cbar.ax.tick_params(labelsize=28)
plt.ylim((c['lat_min'], c['lat_max']))
plt.xlim((c['lon_min'], c['lon_max']))
plt.axis('off')
plt.savefig('results/heatmap-{}-{}.pdf'.format('house-price', c['city_name']), bbox_inches='tight')
plt.clf()
plt.close()

city.plot(alpha=.5, figsize=(10,10), color='gray')
plt.scatter(checkin.lon, checkin.lat, s=checkin.checkins/scaler, alpha=.5, c='blue')
plt.ylim((c['lat_min'], c['lat_max']))
plt.xlim((c['lon_min'], c['lon_max']))
plt.axis('off')
plt.savefig('results/heatmap-{}-{}.pdf'.format('check-ins', c['city_name']), bbox_inches='tight')

