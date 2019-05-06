from grid.create_grid import RegionGrid
import os
import sys
import torch
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config import get_config
from model.utils import write_embeddings
from model.AutoEncoder import AutoEncoder
import numpy as np

if len(sys.argv) > 1:
    epochs = int(sys.argv[1])
    learning_rate = float(sys.argv[2])
    batch_size = int(sys.argv[3])
else:
    epochs = 25
    learning_rate = .1
    batch_size = 20

c = get_config()
region_grid = RegionGrid(config=c)
region_grid.load_img_data(std_img=True)

img_tensor = torch.Tensor(region_grid.img_tensor)
h_dim_size = int(c['hidden_dim_size'])

auto_encoder = AutoEncoder(img_dims=(50, 50), h_dim_size=h_dim_size)
embedding = auto_encoder.run_train_job(n_epoch=epochs, batch_size=batch_size, img_tensor=img_tensor,
                                       lr=learning_rate)

if torch.cuda.is_available():
    embedding = embedding.data.cpu().numpy()
else:
    embedding = embedding.data.numpy()

write_embeddings(arr=embedding, n_nodes=region_grid.n_regions, fname=c['autoencoder_embedding_file'])