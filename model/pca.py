from sklearn.decomposition import PCA
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config import get_config
from grid.create_grid import RegionGrid
from model.utils import write_embeddings


c = get_config()
region_grid = RegionGrid(config=c)
region_grid.load_img_data(std_img=True)
region_grid.load_img_data()

h_dim_size = int(c['hidden_dim_size'])

img_dims = region_grid.img_tensor.shape
n_samples = img_dims[0]
n_channel = img_dims[1]
n_rows = img_dims[2]
n_cols = img_dims[3]
flat_shape = n_channel*n_rows*n_cols
X = region_grid.img_tensor.reshape((n_samples, flat_shape))


pca_model = PCA(n_components=h_dim_size)

embedding = pca_model.fit_transform(X)



write_embeddings(embedding, n_nodes=n_samples, fname=c['pca_file'])