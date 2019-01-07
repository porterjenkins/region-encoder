# region-representation-learning
Learning representations of geo-spatial regions


## Config File

The project uses a config file (called config.json) to do basic setup (e.g., pointing to data set location). Currently, the config file must have the following parameters:

``` 	

{
	"data_dir_main": "foo/bar/city/",
	"census_income_file": "Household Income by Race and Census Tract and Community Area.xlsx",
	"tract_shape_file": "Census-Tracts-2010/chicago-tract.shp",
	"poi_file": "all_POIs_chicago.p",
	"poi_dist_file": "POI_dist_tract.csv",
	"edge_list_file": "edge_list.txt",
	"deepwalk_file": "deepwalk_embedding.txt",
	"embedding_file": "embedding.txt",
	"raw_flow_file": "taxi-trips.csv",
	"flow_mtx_file": "flow_mtx.p",
	"housing_data_file": "zillow_house_price.csv",
	"lon_min": "-87.6998",
	"lon_max": "-87.6065",
	"lat_min": "41.8542",
	"lat_max": "41.9013",
	"grid_size": "5",
	"g_maps_key": "xxxxyyyxxxx",
	"path_to_image_dir": "foo/bar/city/images_no_marker"
}
``` 
