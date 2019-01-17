import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from model.RegionEncoder import RegionEncoder
from grid.create_grid import RegionGrid, get_images_for_grid
from config import get_config
import pickle

# Main script if region area, or grid configuration changes

c = get_config()
# init region
region_grid = RegionGrid(config=c)
print("Initializing Region: Lat [{}, {}], Lon: [{}, {}]".format(region_grid.lat_min, region_grid.lat_max,
                                                                region_grid.lon_min, region_grid.lat_max))
# pull images
get_images_for_grid(region_grid, clear_dir=True)
# load images
region_grid.load_img_data(std_img=True)
# Compute weighted edge matrix W
W = region_grid.create_flow_matrix(c['raw_flow_file'], region_name=c['city_name'])
with open(c['flow_mtx_file'], 'wb') as f:
    pickle.dump(W, f)

region_grid.load_weighted_mtx()
region_grid.load_img_data()

# hyperparameters
n_nodes = len(region_grid.regions)
n_nodal_features = region_grid.feature_matrix.shape[1]
h_dim_graph = 64
h_dim_img = 64
h_dim_size = int(c['hidden_dim_size'])
lambda_ae = .5
lambda_edge = 0.05
lambda_g = 1.0
neg_samples_gcn = 10
epochs = 50
learning_rate = .1

if len(sys.argv) > 1:
    epochs = int(sys.argv[1])
    learning_rate = float(sys.argv[2])

mod = RegionEncoder(n_nodes=n_nodes,
                    n_nodal_features=n_nodal_features,
                    h_dim_graph=h_dim_graph,
                    h_dim_img=h_dim_img,
                    lambda_ae=lambda_ae,
                    lambda_edge=lambda_edge,
                    lambda_g=lambda_g,
                    neg_samples_gcn=neg_samples_gcn,
                    h_dim_size=h_dim_size)
mod.run_train_job(region_grid, epochs=epochs, lr=learning_rate, tol_order=3)