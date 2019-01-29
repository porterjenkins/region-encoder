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
from sklearn.neighbors import NearestNeighbors



class MSNEAutoEncoder(nn.Module):
    def __init__(self, network_size, hidden_dim):
        super(MSNEAutoEncoder, self).__init__()
        ### Encoder
        self.W_encoder_1 = nn.Linear(network_size, 256)
        self.W_encoder_2 = nn.Linear(256, 84)
        self.W_encoder_3 = nn.Linear(84, hidden_dim)

        ### Decoder
        self.W_decoder_1 = nn.Linear(hidden_dim, 256)
        self.W_decoder_2 = nn.Linear(256, network_size)

    def get_q_star(self,Q, H, top_k):
        #q_0 = Q[0, :]

        #q = torch.einsum('ij, k-> ijk', Q, H)
        ##q = torch.mul(q_0, H)

        Q_star = torch.zeros_like(H)
        for i in range(Q.shape[0]):
            q = 0
            nbr_idx = top_k[i]
            for j in nbr_idx:

                q += Q[i,j]*H[j, :]

            Q_star[i, :] = q



        return Q_star


    def forward(self, X, Q, top_k):

        # Encode
        H = F.relu(self.W_encoder_1(X))
        H = F.relu(self.W_encoder_2(H))
        H = F.relu(self.W_encoder_3(H))
        Q_star = self.get_q_star(Q, H, top_k)
        Z = H + Q_star

        # Decode
        H = F.relu(self.W_decoder_1(Z))
        X_reconstruction = F.relu(self.W_decoder_2(H))

        return X_reconstruction, Z


class MSNE(nn.Module):
    def __init__(self, network_size, hidden_dim):
        super(MSNE, self).__init__()
        self.dist_encoder = MSNEAutoEncoder(network_size, hidden_dim)
        self.mobility_encoder = MSNEAutoEncoder(network_size, hidden_dim)

        self.hidden_dim = hidden_dim
        self.network_size = network_size



    def forward(self, X_dist, X_mobility, Q_dist, Q_mobility, k_dist, k_mobility):
        X_dist_hat, H_dist = self.dist_encoder.forward(X_dist, Q_dist, k_dist)
        X_mobility_hat, H_mobility = self.mobility_encoder.forward(X_mobility, Q_mobility, k_mobility)

        return X_dist_hat, H_dist, X_mobility_hat, H_mobility



    def get_optimizer(self, lr):
        optimizer = optim.SGD(self.parameters(), lr=lr, momentum=0.9)
        self.mse_loss = torch.nn.MSELoss(reduction='mean')

        return optimizer


    def loss(self, X_dist_true, X_dist_hat, X_mobility_true, X_mobility_hat):

        #mse_dist = self.mse_loss(X_dist_hat, X_dist_true)
        #mse_mobility = self.mse_loss(X_mobility_hat, X_mobility_true)

        err_dist = X_dist_true - X_dist_hat
        mse_dist = torch.mean(torch.pow(err_dist, 2))

        err_mobility = X_mobility_true - X_mobility_hat
        mse_mobility = torch.mean(torch.pow(err_mobility, 2))

        return mse_dist + mse_mobility


    def run_train_job(self, X_poi_dist, X_poi_mobility, Q_dist, Q_mobility, k_dist, k_mobility, n_epochs, lr):

        n_samples = X_poi_dist.shape[0]
        optimizer = self.get_optimizer(lr)

        H_global = torch.zeros((n_samples, self.hidden_dim*2))

        print("Beginning Train Job for MSNE - epochs: {}, lr: {}".format(n_epochs, lr))
        for e in range(n_epochs):
            permute_idx = np.random.permutation(np.arange(n_samples))
            #running_loss = 0
            #for step in range(int(n_samples / batch_size)):
            # zero the parameter gradients
            optimizer.zero_grad()
            #start_idx = step * batch_size
            #end_idx = start_idx + batch_size
            #batch_idx = permute_idx[start_idx:end_idx]

            #Q_dist_batch = Q_dist[batch_idx, :]
            #Q_dist_batch = Q_dist_batch[:, batch_idx]

            #Q_mobility_batch = Q_mobility[batch_idx, :]
            #Q_mobility_batch = Q_mobility_batch[:, batch_idx]

            X_hat_dist, H_dist, X_hat_mobility, H_mobility = self.forward(X_poi_mobility, X_poi_dist, Q_dist, Q_mobility, k_dist, k_mobility)

            loss = self.loss(X_poi_dist, X_hat_dist, X_poi_mobility, X_hat_mobility)
            loss.backward()
            optimizer.step()

            #running_loss += loss.item()
            #print("--> Step {} train loss: {:.4f}".format(step+1,loss.item()))

            # store updated hidden state
            H_global[:, :self.hidden_dim] = H_dist
            H_global[:, (self.hidden_dim):self.hidden_dim*2] = H_mobility


            #avg_loss = running_loss/int(n_samples / batch_size)
            avg_loss = loss.item()
            print("Epoch: {} - Train loss: {:.6f}".format(e+1, avg_loss))

        return H_global




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


def get_top_k(X, k):

    d = {}
    for i in range(X.shape[0]):
        r_i_corr = np.abs(X[i, :]).argsort()
        top_k = r_i_corr[-(k+1):]

        d[i] = top_k[top_k != i][:k]

    return d



if __name__ == "__main__":

    c = get_config()
    region_grid = RegionGrid(config=c)

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

    #normalize data
    X_poi_dist = region_grid.normalize_mtx(X_poi_dist)
    X_poi_mobility = region_grid.normalize_mtx(X_poi_mobility)

    # Get Q - Autocorrelaton matrix
    Q_dist = np.nan_to_num(np.corrcoef(X_poi_dist), 0)
    Q_mobility = np.nan_to_num(np.corrcoef(X_poi_mobility), 0)

    k_dist = get_top_k(Q_dist, k=5)
    k_mobility = get_top_k(Q_mobility, k=5)



    network_size = X_poi_dist.shape[1]
    msne = MSNE(network_size, hidden_dim=int(int(c['hidden_dim_size'])/2))

    X_poi_dist = torch.Tensor(X_poi_dist)
    X_poi_mobility = torch.Tensor(X_poi_dist)

    if int(sys.argv[1]) > 1:
        n_epochs = sys.argv[1]
        lr = sys.argv[2]
    else:
        n_epochs = 10
        lr = .5

    embedding = msne.run_train_job(X_poi_dist, X_poi_mobility, Q_dist, Q_mobility, k_dist, k_mobility,
                                   n_epochs=n_epochs, lr=lr)

    if torch.cuda.is_available():
        embedding = embedding.data.cpu().numpy()
    else:
        embedding = embedding.data.numpy()

    write_embeddings(arr=embedding, n_nodes=region_grid.n_regions, fname=c['msne_file'])
