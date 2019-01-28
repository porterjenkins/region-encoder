import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config import get_config
from grid.create_grid import RegionGrid
from model.RegionEncoder import RegionEncoder
from model.utils import write_embeddings
import torch
import numpy as np

if len(sys.argv) > 1:
    N_RUNS = int(sys.argv[1])
else:
    N_RUNS = 15


c = get_config()
region_grid = RegionGrid(config=c)
region_grid.load_img_data(std_img=True)
region_grid.load_weighted_mtx()

OUT_DIR = c['data_dir_main'] + c['hyperparams_dir'] + "/"

# hyperparameters
n_nodes = len(region_grid.regions)
n_nodal_features = region_grid.feature_matrix.shape[1]
h_dim_graph = 64
h_dim_img = 32
h_dim_size = int(c['hidden_dim_size'])
context_gcn = 4
neg_samples_gcn = 10
epochs = 12
learning_rate = .1
img_dims = (50, 50)


params = {}
for i in range(N_RUNS):
    params[i] = []
    # params to tune
    lambda_ae = np.exp(np.random.uniform(-6, .6))
    lambda_edge = np.exp(np.random.uniform(-6, .5))
    lambda_g = np.exp(np.random.uniform(-6, .5))
    lambda_weight_decay = np.random.uniform(0, 1e-3)

    print(">>>> Tuning Iteration: {} - {:.4f}, {:.4f}, {:.4f}, {:.4f} <<<<".format(i+1, lambda_ae, lambda_edge, lambda_g,
                                                                         lambda_weight_decay))

    params[i].append(lambda_ae)
    params[i].append(lambda_edge)
    params[i].append(lambda_g)
    params[i].append(lambda_weight_decay)




    mod = RegionEncoder(n_nodes=n_nodes,
                            n_nodal_features=n_nodal_features,
                            h_dim_graph=h_dim_graph,
                            h_dim_img=h_dim_img,
                            lambda_ae=lambda_ae,
                            lambda_edge=lambda_edge,
                            lambda_g=lambda_g,
                            neg_samples_gcn=neg_samples_gcn,
                            h_dim_size=h_dim_size,
                            img_dims=img_dims,
                            lambda_weight_decay=lambda_weight_decay)
    mod.run_train_job(region_grid, epochs=epochs, lr=learning_rate, tol_order=3)

    if torch.cuda.is_available():
        embedding = mod.embedding.data.cpu().numpy()
    else:
        embedding = mod.embedding.data.numpy()




    fname = "{}embedding-iter-{}.txt".format(OUT_DIR, i)
    write_embeddings(arr=embedding, n_nodes=n_nodes, fname=fname)
    mod.plt_learning_curve("plots/region-learning-curve.pdf", plt_all=False, log_scale=False)

    with open(OUT_DIR + "params-{}.txt".format(i), 'w') as f:
        param = params[i]
        f.write("{}: ".format(i))
        for p in param:
            f.write("{:.4f}, ".format(p))

        f.write("\n")