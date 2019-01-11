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
# pull images
get_images_for_grid(region_grid, clear_dir=True)
# load images
region_grid.load_img_data(std_img=True)
# Compute weighted edge matrix W
W = region_grid.create_flow_matrix(c['raw_flow_file'])
with open(c['flow_mtx_file'], 'wb') as f:
    pickle.dump(W, f)

region_grid.load_weighted_mtx()
region_grid.load_img_data()

# Run Train job
n_nodes = len(region_grid.regions)
mod = RegionEncoder(n_nodes=n_nodes, n_nodal_features=552, h_dim_graph=64, lambda_ae=.5, lambda_edge=.1,
                    lambda_g=0.05, neg_samples_gcn=25)

mod.run_train_job(region_grid, epochs=100, lr=.01, tol_order=3)
mod.write_embeddings(c['embedding_file'])
mod.plt_learning_curve("plots/region-learning-curve.pdf")