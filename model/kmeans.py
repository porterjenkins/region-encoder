import numpy as np
from sklearn.cluster import KMeans
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config import get_config
from grid.create_grid import RegionGrid
from model.utils import write_embeddings

c = get_config()
region_grid = RegionGrid(config=c)
region_grid.load_img_data(std_img=True)

img_tensor_dims = region_grid.img_tensor.shape

I = region_grid.img_tensor.reshape(img_tensor_dims[0], img_tensor_dims[1]*img_tensor_dims[2]*img_tensor_dims[3])

k_means = KMeans(n_clusters=int(c['hidden_dim_size']))

embedding = k_means.fit_transform(X=I)

write_embeddings(embedding, n_nodes=region_grid.n_regions, fname=c['kmeans_file'])



