import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config import get_config
from grid.create_grid import RegionGrid
import numpy as np
import matplotlib.pyplot as plt
#os.environ['PROJ_LIB'] = "//Users/porterjenkins/anaconda/envs/py3-region-rep/share/proj/"
#from mpl_toolkits.basemap import Basemap
import geopandas as gpd
import pandas as pd
import descartes
from shapely.geometry import Point, Polygon

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




#shape_fname = c['data_dir_main'] + 'geo_export_f0db6423-ef91-4ee9-b18a-7b56cf970085.shp'


city = gpd.read_file(c['shape_file'])
city.plot(alpha=.5, figsize=(10, 10), color='gray')

x = house_price.lon
y = house_price.lat
z = house_price['priceSqft']

cm = plt.cm.get_cmap('coolwarm')
sc = plt.scatter(x, y, c=z, cmap=cm, alpha=.4,s=4)

#plt.ylim((c['lat_min'], c['lat_max']))
#plt.xlim((c['lon_min'], c['lon_max']))
plt.colorbar(sc)
plt.ylim((c['lat_min'], c['lat_max']))
plt.xlim((c['lon_min'], c['lon_max']))
plt.axis('off')
plt.savefig('results/heatmap-{}-{}.pdf'.format('house-price', c['city_name']), bbox_inches='tight')
plt.clf()
plt.close()

city.plot(alpha=.5, figsize=(10,10), color='gray')
if c["city_name"] == 'nyc':
    scaler = 50
else:
    scaler = 10
plt.scatter(checkin.lon, checkin.lat, s=checkin.checkins/scaler, alpha=.75, c='orange')
plt.ylim((c['lat_min'], c['lat_max']))
plt.xlim((c['lon_min'], c['lon_max']))
plt.axis('off')
plt.savefig('results/heatmap-{}-{}.pdf'.format('check-ins', c['city_name']), bbox_inches='tight')

