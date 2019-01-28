from sklearn.decomposition import NMF
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config import get_config
from grid.create_grid import RegionGrid
from model.utils import write_embeddings

if len(sys.argv) == 1:
    raise Exception("No input. Input must be 'A' (adjacency) or 'W' (weighted)")

c = get_config()
region_grid = RegionGrid(config=c)
region_grid.load_img_data(std_img=True)
region_grid.load_weighted_mtx()

A = region_grid.adj_matrix
W = region_grid.weighted_mtx
h_dim_size = int(c['hidden_dim_size'])

mf_model = NMF(n_components=h_dim_size, init='random', solver='cd', max_iter=500, l1_ratio=0.0)



if sys.argv[1] == "A":
    print("Factorizing Adjacency Matrix - Hidden Size: {}".format(h_dim_size))
    embedding = mf_model.fit_transform(A)
    n_nodes = A.shape[0]
elif sys.argv[1] == "W":
    print("Factorizing Weighted (flow) Matrix - Hidden Size: {}".format(h_dim_size))
    embedding = mf_model.fit_transform(W)
    n_nodes = W.shape[0]
else:
    raise NotImplementedError("Input must be 'A' (adjacency) or 'W' (weighted)")


write_embeddings(embedding, n_nodes=n_nodes, fname=c['nmf_file'])

