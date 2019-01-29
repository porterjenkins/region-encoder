from sklearn.decomposition import NMF
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config import get_config
from grid.create_grid import RegionGrid
from model.utils import write_embeddings
import numpy as np
from geopy.distance import distance
import pickle
from scipy import sparse
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


class MSNE(nn.Module):
    def __init__(self, network_size, hidden_dim):
        super(MSNE, self).__init__()
        ### Encoder
        self.W_encoder_1 = nn.Linear(network_size, 256)
        self.W_encoder_2 = nn.Linear(256, 84)
        self.W_encoder_3 = nn.Linear(84, hidden_dim)

        ### Decoder
        self.W_decoder_1 = nn.Linear(hidden_dim, 256)
        self.W_decoder_2 = nn.Linear(256, network_size)




    def forward(self, X):

        # Encode
        H = F.relu(self.W_encoder_1(X))
        H = F.relu(self.W_encoder_2(H))
        Z = F.relu(self.W_encoder_3(H))

        # Decode
        H = F.relu(self.W_decoder_1(Z))
        X_reconstruction = F.relu(self.W_decoder_2(H))

        return X_reconstruction


    def get_optimizer(self, lr):
        optimizer = optim.SGD(self.parameters(), lr=lr, momentum=0.9)
        self.mse_loss = torch.nn.MSELoss(reduction='mean')

        return optimizer


    def loss(self, X_dist_true, X_dist_hat, X_mobility_true, X_mobility_hat):

        mse_dist = self.mse_loss(X_dist_hat, X_dist_true)
        mse_mobility = self.mse_loss(X_mobility_hat, X_mobility_true)

        return mse_dist + mse_mobility


    def run_train_job(self, X_poi_dist, X_poi_mobility, n_epochs, batch_size, lr):

        n_samples = X_poi_dist.shape[0]
        optimizer = self.get_optimizer(lr)

        print("Beginning Train Job for MSNE - epochs: {}, batch size: {}, lr: {}".format(n_epochs, batch_size, lr))
        for e in range(n_epochs):
            permute_idx = np.random.permutation(np.arange(n_samples))
            running_loss = 0
            for step in range(int(n_samples / batch_size)):
                # zero the parameter gradients
                optimizer.zero_grad()
                start_idx = step * batch_size
                end_idx = start_idx + batch_size
                batch_idx = permute_idx[start_idx:end_idx]

                X_hat_dist = self.forward(X_poi_dist[batch_idx, :])
                X_hat_mobility = self.forward(X_poi_mobility[batch_idx, :])

                loss = self.loss(X_poi_dist[batch_idx, :], X_hat_dist, X_poi_mobility[batch_idx, :], X_hat_mobility)
                loss.backward()

                running_loss += loss.item()

            avg_loss = running_loss/int(n_samples / batch_size)
            print("Epoch: {} - Average Train loss: {:.4f}".format(e+1, avg_loss))



def get_poi_poi_dist_mtx(region_grid, n_region, n_cat):
    X = np.zeros((n_region, n_cat ** 2))
    cntr = 0
    print("Getting Intra-region POI-POI distance networks")
    for r in region_grid.regions.values():
        print("--> progress: {:.4f}".format(cntr / region_grid.n_regions), end='\r')
        x = r.get_poi_poi_dist(region_grid.categories)
        X[r.index, :] = x.flatten()
        cntr += 1

    return sparse.csr_matrix(X)

def get_poi_poi_mobility_mtx(region_grid):
    print("Getting Intra-region POI-POI mobility networks")
    n_cat = len(region_grid.categories)
    region_cntr = 0
    X = np.zeros((region_grid.n_regions, n_cat ** 2))
    for r in region_grid.regions.values():
        poi_poi_network = np.zeros((n_cat, n_cat))
        n_trips = len(r.trips)
        for trip_cnt, t in enumerate(r.trips):
            print("Progress - Regions: {:.4f}, Trips: {:.4f}".format(region_cntr/region_grid.n_regions,
                                                                     trip_cnt/n_trips), end='\r')
            pick_up_dist = np.zeros(len(r.poi))
            drop_off_dist = np.zeros(len(r.poi))

            for i, p in enumerate(r.poi):
                t_pick_lat = t[0]
                t_pick_lon = t[1]
                t_drop_lat = t[2]
                t_drop_lon = t[3]

                pick_up_dist[i] = distance((t_pick_lat, t_pick_lon), (p.location.lat, p.location.lon)).m
                drop_off_dist[i] = distance((t_drop_lat, t_drop_lon), (p.location.lat, p.location.lon)).m

            try:
                poi_pick_idx = np.argmin(pick_up_dist)
                poi_drop_idx = np.argmin(drop_off_dist)

                pick_cat = r.poi[poi_pick_idx].cat
                drop_cat = r.poi[poi_drop_idx].cat

                poi_poi_network[region_grid.categories[pick_cat], region_grid.categories[drop_cat]] += 1
            except ValueError:
                pass

        X[r.index, :] = poi_poi_network.flatten()
        region_cntr +=1


    return sparse.csr_matrix(X)




if __name__ == "__main__":
    c = get_config()
    region_grid = RegionGrid(config=c)

    if int(sys.argv[1]) == 1:
        GET_POI_FLAG = True
    else:
        GET_POI_FLAG = False

    if GET_POI_FLAG:

        # get intra-region poi-poi distancences
        n_categories = len(region_grid.categories)
        X_poi_dist = get_poi_poi_dist_mtx(region_grid, region_grid.n_regions, n_categories)
        with open(c['data_dir_main'] + "mnse_poi_dist_mtx.p", 'wb') as f:
            pickle.dump(X_poi_dist, f)


        # get intra-region poi-poi mobility
        f = c['raw_flow_file'].split(".csv")[0] + "-sampled.csv"
        region_grid.get_taxi_trips(f)
        X_poi_mobility = get_poi_poi_mobility_mtx(region_grid)

        with open(c['data_dir_main'] + "mnse_poi_mobility_mtx.p", 'wb') as f:
            pickle.dump(X_poi_mobility, f)


    else:

        with open(c['data_dir_main'] + "mnse_poi_dist_mtx.p", 'rb') as f:
            X_poi_dist = pickle.load(f)
            X_poi_dist = X_poi_dist.todense()

        with open(c['data_dir_main'] + "mnse_poi_mobility_mtx.p", 'rb') as f:
            X_poi_mobility = pickle.load(f)
            X_poi_mobility = X_poi_mobility.todense()


    network_size = X_poi_dist.shape[1]
    msne = MSNE(network_size, hidden_dim=int(c['hidden_dim_size']))

    X_poi_dist = torch.Tensor(X_poi_dist)
    X_poi_mobility = torch.Tensor(X_poi_dist)

    msne.run_train_job(X_poi_dist, X_poi_mobility, n_epochs=5, batch_size=50, lr=.05)
