import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from config import get_config
from model.utils import load_embedding, write_embeddings
import numpy as np

c = get_config()
grid_size = int(c['grid_size'])
autoencoder_embed_fname = c['autoencoder_embedding_file']
gcn_all_embed_fname = '{}gcn_all_embedding.txt'.format(c['data_dir_main'])


#gcn_embed = load_embedding(gcn_all_embed_fname)
autoencoder_embed = load_embedding(autoencoder_embed_fname)

embed_global = np.concatenate((autoencoder_embed, autoencoder_embed), axis=1)


fname='{}concat_global_embedding.txt'.format(c['data_dir_main'])
write_embeddings(arr=embed_global, n_nodes=grid_size**2, fname=fname)