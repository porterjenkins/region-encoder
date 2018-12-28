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
	"raw_flow_file": "taxi-trips.csv",
	"flow_mtx_file": "flow_mtx.p",
	"lon_min": "41.65021997246178",
	"lon_max": "42.02126162051242",
	"lat_min": "-87.90448852338",
	"lat_max": "-87.53049651540705",
	"g_maps_key": "xxxxyyyxxxx",
	"path_to_image_dir": "foo/bar/city/images_no_marker"
}
``` 
