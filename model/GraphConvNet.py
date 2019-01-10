import torch
import torch.nn as nn
import torch.nn.functional as F
from model.get_karate_data import *
import torch.optim as optim
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config import get_config
from grid.create_grid import RegionGrid


class GCN(nn.Module):
    """
    Graph ConvNet Model (via Kipf and Welling ICLR'2017)
    Three matrices required:
        - A: Adjacency matrix (A + I)
        - D: Diagonal matrix (D^-1/2)
        - X: Nodal feature matrix
    """
    def __init__(self, n_nodes, n_features, h_dim_size=16):
        super(GCN, self).__init__()
        self.n_nodes = n_nodes
        self.n_features = n_features
        # fully connected layer 1
        self.fcl_0 = nn.Linear(n_features, 512, bias=True)
        # Output layer for link prediction
        self.fcl_1 = nn.Linear(512, h_dim_size, bias=True)



    def forward(self, X, A, D):
        # G = D*A*D*X*W
        G_0 = torch.mm(torch.mm(D, torch.mm(A,D)), X)
        H_0 = F.relu(self.fcl_0(G_0))

        G_1 = torch.mm(torch.mm(D, torch.mm(A,D)), H_0)
        H_1 = self.fcl_1(G_1)

        return H_1


    def get_optimizer(self, lr):
        optimizer = optim.SGD(self.parameters(), lr=lr, momentum=0.9)

        return optimizer

    @staticmethod
    def preprocess_adj(A):
        n = A.shape[0]

        return A + np.eye(n)

    @staticmethod
    def preprocess_degree(D):
        # get D^-1/2
        D = np.linalg.inv(D)
        D = np.power(D, .5)

        return D

    @staticmethod
    def gen_pos_samples_gcn(regions, idx_map, h_graph, batch_size):
        pos_sample_map = dict()
        n_h_dims = h_graph.shape[1]
        pos_samples = torch.zeros((batch_size, n_h_dims), dtype=torch.float)

        for id, mtx_idx in idx_map.items():
            region_i = regions[id]
            pos_sample = np.random.choice(list(region_i.adjacent.keys()))
            pos_sample_map[mtx_idx] = idx_map[pos_sample]
            pos_samples[mtx_idx, :] = h_graph[idx_map[pos_sample], :]

        return pos_samples

    @staticmethod
    def gen_neg_samples_gcn(n_neg_samples, adj_mtx, h_graph, idx_map, batch_size):
        neg_sample_map = dict()
        n_h_dims = h_graph.shape[1]
        neg_samples = torch.zeros((batch_size, n_neg_samples, n_h_dims), dtype=torch.float)

        for id, mtx_idx in idx_map.items():
            neg_sample_map[mtx_idx] = list()
            for k in range(n_neg_samples):
                get_neg_sample = True
                while get_neg_sample:
                    neg_sample_idx = np.random.randint(0, batch_size)
                    if adj_mtx[mtx_idx, neg_sample_idx] == 0:
                        get_neg_sample = False
                        neg_sample_map[mtx_idx].append(neg_sample_idx)
                        neg_samples[mtx_idx, k, :] = h_graph[neg_sample_idx, :]

        return neg_samples

    @staticmethod
    def loss_graph(h_graph, pos_samples, neg_samples):
        n = h_graph.shape[0]
        h_dim = h_graph.shape[1]
        n_neg_samples = neg_samples.shape[1]

        h_graph_expanded = h_graph.unsqueeze(1)
        h_graph_expanded = h_graph_expanded.expand(n, n_neg_samples, h_dim)

        neg_dot = -torch.sum(torch.mul(h_graph_expanded, neg_samples), dim=-1)
        neg_dot_sig = torch.sigmoid(neg_dot)
        l_neg_samples_ind = torch.log(neg_dot_sig)
        l_neg_samples_total = torch.sum(l_neg_samples_ind, dim=-1)


        dot = torch.sum(torch.mul(h_graph, pos_samples), dim=-1)
        l_pos_samples = torch.log(dot)

        total_loss = l_pos_samples + l_neg_samples_total
        l_graph = torch.mean(total_loss)

        # to minimize negative log likelihood: multiply by -1
        return -l_graph

    def run_train_job(self, region_grid, n_epoch, n_neg_samples=15, learning_rate=.01):
        optimizer = self.get_optimizer(learning_rate)

        A = region_grid.adj_matrix
        D = region_grid.degree_matrix
        X = region_grid.feature_matrix
        # preprocess step for graph matrices
        A_hat = GCN.preprocess_adj(A)
        D_hat = GCN.preprocess_degree(D)

        region_mtx_map = region_grid.matrix_idx_map
        batch_size = A.shape[0]

        # Cast matrices to torch.tensor
        A_hat = torch.from_numpy(A_hat).type(torch.FloatTensor)
        D_hat = torch.from_numpy(D_hat).type(torch.FloatTensor)
        X = torch.from_numpy(X).type(torch.FloatTensor)

        for epoch in range(n_epoch):  # loop over the dataset multiple times

            # zero the parameter gradients
            optimizer.zero_grad()


            # forward + backward + optimize
            H = self.forward(X=X, A=A_hat, D=D_hat)
            # generate positive samples for gcn
            gcn_pos_samples = GCN.gen_pos_samples_gcn(region_grid.regions, region_mtx_map, H, batch_size)
            # generate negative samples for gcn
            gcn_neg_samples = GCN.gen_neg_samples_gcn(n_neg_samples, A, H, region_mtx_map, batch_size)
            # compute loss
            loss = GCN.loss_graph(H, gcn_pos_samples, gcn_neg_samples)

            loss.backward()
            optimizer.step()

            # print statistics
            print("Epoch: {}, Train Loss {:.4f}".format(epoch, loss))

        print('Finished Training')




if __name__ == "__main__":
    c = get_config()
    region_grid = RegionGrid(config=c, load_imgs=True)
    n_nodes = len(region_grid.regions)
    gcn = GCN(n_nodes=n_nodes, n_features=552, h_dim_size=16)

    gcn.run_train_job(region_grid, n_epoch=100, learning_rate=.01)