import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config import get_config
from grid.create_grid import RegionGrid


class GCN(nn.Module):
    """
    Graph ConvNet Model (via Kipf and Welling ICLR'2017)
    Two matrices required:
        - X: Nodal feature matrix that has been normalized
        - A: Adjacency matrix
    """

    def __init__(self, n_features, h_dim_size=16):

        super(GCN, self).__init__()
        self.n_features = n_features
        self.n_classes = n_features
        self.n_hidden = h_dim_size

        # fully connected layer 1
        self.fcl_0 = nn.Linear(n_features, 100, bias=True)
        # Output layer for link prediction
        self.fcl_1 = nn.Linear(100, h_dim_size, bias=True)

        if torch.cuda.is_available():
            self.fcl_0 = self.fcl_0.cuda()
            self.fcl_1 = self.fcl_1.cuda()

    def forward(self, X):
        a = self.adj

        G_0 = torch.mm(a, X)  # conv 1
        G_0 = self.fcl_0(G_0)  # linear layer 1
        H_0 = F.relu(G_0)
        x = torch.mm(a, H_0)  # conv 2
        return self.fcl_1(x)  # output layer

    def get_optimizer(self, lr):
        optimizer = torch.optim.SGD(self.parameters(), lr=lr, momentum=0.9)

        return optimizer

    @staticmethod
    def get_weighted_proximity(h_graph):
        """
        compute sigmoid of similarity matrix. e.g.,:
            1/(1 + exp(-X))
                where X = HH'
                and H = is an nxp matrix of node represntations
                (each row i is an embedding vector for node i)
        :param h_graph:
        :return:
        """
        h_graph_dissim = -torch.mm(h_graph, torch.transpose(h_graph, 0, 1))
        f_o_proximity = torch.sigmoid(h_graph_dissim)
        return f_o_proximity

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
    def get_neg_sample_distribution(D, adj_list):
        D_diag = D.diagonal()
        no_adj_list = np.invert(adj_list) + 2
        probs = np.multiply(D_diag, no_adj_list)
        probs = probs / probs.sum()
        return probs

    @staticmethod
    def gen_neg_samples_gcn(n_neg_samples, adj_mtx, h_graph, idx_map, batch_size, probs):
        neg_sample_map = dict()
        n_h_dims = h_graph.shape[1]
        neg_samples = torch.zeros((batch_size, n_neg_samples, n_h_dims), dtype=torch.float)
        neg_sample_probs = torch.zeros((batch_size, n_neg_samples), dtype=torch.float)

        for id, mtx_idx in idx_map.items():
            neg_sample_map[mtx_idx] = list()
            for k in range(n_neg_samples):
                get_neg_sample = True
                while get_neg_sample:
                    neg_sample_idx = np.random.choice(np.arange(0, batch_size), p=probs)
                    if adj_mtx[mtx_idx, neg_sample_idx] == 0:
                        # sample node (index) is a true negative sample, then keep
                        get_neg_sample = False
                        neg_sample_map[mtx_idx].append(neg_sample_idx)
                        neg_samples[mtx_idx, k, :] = h_graph[neg_sample_idx, :]
                        neg_sample_probs[mtx_idx, k] = probs[neg_sample_idx]

        return neg_samples, neg_sample_probs

    @staticmethod
    def gen_skip_gram_samples(context_size, n_neg_samples, h_graph, batch_size, idx_map, regions, adj_mtx, degree):
        # generate context (positive samples) and negative sampples for each sample
        n_h_dims = h_graph.shape[1]
        pos_samples = torch.zeros((batch_size, context_size, n_h_dims), dtype=torch.float)
        neg_samples = torch.zeros((batch_size, context_size, n_neg_samples, n_h_dims), dtype=torch.float)
        sampled_probs = torch.zeros((batch_size, context_size,  n_neg_samples), dtype=torch.float)

        for id, mtx_idx in idx_map.items():

            # create context by sampling adjacent nodes WITH REPLACEMENT
            adj_nodes = list(regions[id].adjacent.keys())
            context = np.random.choice(adj_nodes, replace=True, size=context_size)
            for c in range(context_size):
                pos_samples[mtx_idx, c, :] = h_graph[idx_map[context[c]], :]

                probs = GCN.get_neg_sample_distribution(degree, adj_mtx[mtx_idx, :])
                neg_samples_idx = np.random.choice(np.arange(0, batch_size), p=probs, size=n_neg_samples)
                for k in range(n_neg_samples):
                    # insert embedding vector to tensor via id lookup

                    neg_samples[mtx_idx, c, k, :] = h_graph[neg_samples_idx[k], :]
                    sampled_probs[mtx_idx, c, k] = probs[neg_samples_idx[k]]

        return pos_samples, neg_samples, sampled_probs

    @staticmethod
    def skip_gram_loss(h_graph, pos_samples, neg_samples, neg_probs):
        if torch.cuda.is_available():
            h_graph = h_graph.cuda()
            pos_samples = pos_samples.cuda()
            neg_samples = neg_samples.cuda()
            neg_probs = neg_probs.cuda()


        n = h_graph.shape[0]
        h_dim = h_graph.shape[1]
        n_neg_samples = neg_samples.shape[2]
        n_pos_samples = pos_samples.shape[1]

        h_graph_expanded_3_dims = h_graph.unsqueeze(1)
        h_graph_expanded_3_dims = h_graph_expanded_3_dims.expand(n, n_pos_samples, h_dim)
        h_graph_expanded_4_dims = h_graph_expanded_3_dims.unsqueeze(2)
        h_graph_expanded_4_dims = h_graph_expanded_4_dims.expand(n, n_pos_samples, n_neg_samples, h_dim)

        neg_dot = -torch.sum(torch.mul(h_graph_expanded_4_dims, neg_samples), dim=-1)
        neg_dot_sig = torch.sigmoid(neg_dot)
        #  Weight negative sample loss contribution by probability
        l_neg_samples_ind = torch.log(neg_dot_sig)
        l_neg_samples_ind_mean = torch.mul(l_neg_samples_ind, neg_probs)
        l_neg_samples_total = torch.sum(l_neg_samples_ind_mean, dim=-1)

        dot = torch.sum(torch.mul(h_graph_expanded_3_dims, pos_samples), dim=-1)
        dot_sig = torch.sigmoid(dot)
        l_pos_samples = torch.log(dot_sig)

        total_loss = torch.sum(l_pos_samples + l_neg_samples_total, dim=-1)
        l_graph = torch.mean(total_loss)

        # to minimize negative log likelihood: multiply by -1
        return -l_graph

    @staticmethod
    def loss_weighted_edges(learned_graph_prox, empirical_graph_prox):
        """
        Compute KL Divergence between learned graph proximity and empircal proximity
        of weighted edges
        :param learned_graph_prox:
        :param empirical_graph_prox:
        :return:
        """
        loss_ind = empirical_graph_prox * torch.log(learned_graph_prox)
        loss_ind_triu = torch.triu(loss_ind)
        loss = - torch.sum(loss_ind_triu)
        return loss

    @staticmethod
    def loss_main(loss_sg, loss_we, penalty=(1.0, 1.0)):

        return penalty[0] * loss_sg + penalty[1] * loss_we

    # ideally  this should be outside the module
    # we should be passing all the parameters it needs then run it as opposed to half in and half out
    def run_train_job(self, region_grid, n_epoch, n_neg_samples=15, n_pos_samples=4,learning_rate=.01, penalty=(1.0, 1.0)):
        optimizer = self.get_optimizer(learning_rate)

        A = region_grid.adj_matrix
        D = region_grid.degree_matrix
        X = region_grid.feature_matrix
        W = region_grid.weighted_mtx
        # preprocess step for graph matrices
        A_hat = GCN.preprocess_adj(A)
        D_hat = GCN.preprocess_degree(D)

        region_mtx_map = region_grid.matrix_idx_map
        batch_size = A.shape[0]

        # Cast matrices to torch.tensor
        A_hat = torch.from_numpy(A_hat).type(torch.FloatTensor)
        D_hat = torch.from_numpy(D_hat).type(torch.FloatTensor)

        X = torch.from_numpy(X).type(torch.FloatTensor)
        W = torch.from_numpy(W).type(torch.FloatTensor)

        # id like to move this out side of the instance
        self.adj = torch.mm(D_hat, torch.mm(A_hat, D_hat))
        if torch.cuda.is_available():
            self.adj = self.adj.cuda()
            X = X.cuda()
            W = W.cuda()

        for epoch in range(n_epoch):  # loop over the dataset multiple times

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward propogation step
            H = self.forward(X=X)

            # Generate context (positive samples) and negative samples for SkipGram Loss
            gcn_pos_samples, gcn_neg_samples, neg_probs = GCN.gen_skip_gram_samples(n_pos_samples, n_neg_samples, H,
                                                                                    batch_size, region_mtx_map,
                                                                                    region_grid.regions, A, D)
            # compute skipgram loss
            loss_skip_gram = GCN.skip_gram_loss(H, gcn_pos_samples, gcn_neg_samples, neg_probs)

            # compute first order proximity loss
            # get learned first-order proximity
            graph_proximity = GCN.get_weighted_proximity(H)
            # normalize empirical proximity over whole matrix W
            emp_proximity = W / torch.sum(W)
            # compute loss
            loss_edge_weights = GCN.loss_weighted_edges(graph_proximity, emp_proximity)

            loss = GCN.loss_main(loss_skip_gram, loss_edge_weights, penalty)
            loss.backward()
            optimizer.step()

            # print statistics
            print("Epoch: {}, Train Loss {:.4f} -- skip-gram: {:.4f}, first-order: {:.4f}".format(epoch, loss,
                                                                                                  loss_skip_gram,
                                                                                                  loss_edge_weights))

        print('Finished Training')


def to_torch_tensor(matrix, to_type=torch.FloatTensor):
    return torch.from_numpy(matrix).type(to_type)


if __name__ == "__main__":
    c = get_config()
    region_grid = RegionGrid(config=c)
    region_grid.load_weighted_mtx()
    n_nodes = len(region_grid.regions)
    gcn = GCN(n_features=552, h_dim_size=32)
    gcn.run_train_job(region_grid, n_epoch=100, learning_rate=.05, penalty=(1, 1), n_neg_samples=15, n_pos_samples=4)
