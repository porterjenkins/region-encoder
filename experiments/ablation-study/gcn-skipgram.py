import os
import sys
import torch
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from config import get_config
from model.utils import write_embeddings
from model.GraphConvNet import GCN
from grid.create_grid import RegionGrid

if len(sys.argv) > 1:
    epochs = int(sys.argv[1])
    learning_rate = float(sys.argv[2])
else:
    epochs = 25
    learning_rate = .1


context_gcn = 4
neg_samples_gcn = 10


c = get_config()
region_grid = RegionGrid(config=c)
region_grid.load_weighted_mtx()

n_nodes = len(region_grid.regions)
h_dim_size = int(c['hidden_dim_size'])
n_nodal_features = region_grid.feature_matrix.shape[1]



gcn = GCN(n_features=n_nodal_features, h_dim_size=h_dim_size)

# set flow mtx reconstruction loss penalty to 0
embedding = gcn.run_train_job(region_grid, n_epoch=epochs, learning_rate=learning_rate, penalty=(1, 0),
                              n_neg_samples=neg_samples_gcn, n_pos_samples=context_gcn)

if torch.cuda.is_available():
    embedding = embedding.data.cpu().numpy()
else:
    embedding = embedding.data.numpy()

embed_fname = '{}gcn_skipgram_embedding.txt'.format(c['data_dir_main'])
write_embeddings(arr=embedding, n_nodes=region_grid.n_regions, fname=embed_fname)
