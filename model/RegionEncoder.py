import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import torch
import torch.nn as nn
import torch.optim as optim
from model.AutoEncoder import AutoEncoder
from model.GraphConvNet import GCN
from model.discriminator import DiscriminatorMLP
from model.get_karate_data import *
from config import get_config

from grid.create_grid import RegionGrid




class RegionEncoder(nn.Module):
    """
    Implementation of proposed model for
    Multi-Modal Region Encoding (MMRE)
    """
    def __init__(self, n_nodes, n_nodal_features, h_dim_graph=4, h_dim_img=32, h_dim_disc=32,
                 lambda_ae=.1, lambda_g=.1, lambda_edge=.1, lambda_weight_decay=.01, img_dims=(640,640)):
        super(RegionEncoder, self).__init__()
        # Model Layers
        self.graph_conv_net = GCN(n_nodes=n_nodes, n_features=n_nodal_features, h_dim_size=h_dim_graph, n_classes=4)
        self.auto_encoder = AutoEncoder(h_dim_size=h_dim_img, img_dims=img_dims)
        self.discriminator = DiscriminatorMLP(x_features=h_dim_graph, z_features=h_dim_img, h_dim_size=h_dim_disc)

        # Model Hyperparams
        self.lambda_ae = lambda_ae
        self.lambda_g = lambda_g
        self.lambda_edge = lambda_edge
        self.lambda_wd = lambda_weight_decay

        # Canned Torch loss objects
        self.cross_entropy = nn.CrossEntropyLoss()
        self.bce_logits = nn.BCEWithLogitsLoss()

    def forward(self, X, A, D, img_tensor):


        # Forward step for graph data
        h_graph = self.graph_conv_net.forward(X, A, D)
        graph_proximity = self._get_weighted_proximity(h_graph)

        # Forward step for image data
        image_hat, h_image = self.auto_encoder.forward(img_tensor)

        # forward step for discriminator (all data)
        logits, h_global = self.discriminator.forward(x=h_graph, z=h_image, activation=False)

        return logits, h_global, image_hat, graph_proximity, h_graph, h_image


    def _get_weighted_proximity(self, h_graph):
        """
        compute sigmoid of similarity matrix. e.g.,:
            1/(1 + exp(-X))
                where X = HH'
                and H = is an nxp matrix of node represntations
                (each row i is an embedding vector for node i)
        :param h_graph:
        :return:
        """
        h_graph_sim = torch.mm(h_graph, torch.transpose(h_graph, 0, 1))
        f_o_proximity = torch.sigmoid(h_graph_sim)
        return f_o_proximity

    def weight_decay(self):
        #w_0 = self.graph_conv_net.fcl_0.weight.data
        #tmp = torch.norm(w_0, p='fro')

        reg = 0


        for p in self.parameters():
            layer = p.data

            reg += self.lambda_wd * torch.norm(layer)

        return reg

    def get_optimizer(self, lr):
        optimizer = optim.SGD(self.parameters(), lr=lr, momentum=0.9)

        return optimizer

    def loss_graph(self, h_graph, pos_samples, neg_samples):
        """

        :param graph_logits:
        :param gamma:
        :return:
        """
        # TODO: rewrite for numerical stability with

        eps = 10e-5

        n = h_graph.shape[0]
        h_dim = h_graph.shape[1]
        n_neg_samples = neg_samples.shape[1]

        h_graph_expanded = h_graph.unsqueeze(1)
        h_graph_expanded = h_graph_expanded.expand(n, n_neg_samples, h_dim)

        neg_dot = -torch.sum(torch.mul(h_graph_expanded, neg_samples), dim=(1,2))
        l_neg_samples = torch.log(torch.clamp(torch.sigmoid(neg_dot), min=eps, max=1-eps))


        dot = torch.sum(torch.mul(h_graph, pos_samples), dim=-1)
        l_pos_samples = torch.log(torch.clamp(torch.sigmoid(dot), min=eps, max=1-eps))

        l_graph = torch.mean(l_pos_samples + l_neg_samples)

        return l_graph

    def loss_disc(self, eta, eta_logits):
        loss_disc = self.bce_logits(eta_logits, eta)
        return loss_disc


    def loss_ae(self, img_input, img_reconstruction):
        err = img_input - img_reconstruction
        mse = torch.mean(torch.pow(err, 2))

        return mse

    def loss_weighted_edges(self, learned_graph_prox, empirical_graph_prox):
        """
        Compute KL Divergence between learned graph proximity and empircal proximity
        of weighted edges
        :param learned_graph_prox:
        :param empirical_graph_prox:
        :return:
        """
        loss_ind = empirical_graph_prox * torch.log(learned_graph_prox)
        loss_ind = torch.triu(loss_ind)
        loss = - torch.sum(loss_ind)

        return loss

    def loss_function(self, L_graph, L_edge_weights, L_disc, L_ae):
        """
        Global loss function for model. Loss has the following components:
            - Reconstruction of spatial graph
            - Prediction of flow graph
            - Reconstruction of image
            - Error of the discriminator
        :param L_graph:
        :param L_edge_weights:
        :param L_disc:
        :param L_ae:
        :return:
        """
        #regularizer = self.weight_decay()

        L = L_disc + self.lambda_ae * L_ae + self.lambda_g * L_graph + self.lambda_edge * L_edge_weights

        return L

    def __preprocess_adj(self, A):
        n = A.shape[0]

        return A + np.eye(n)


    def __preprocess_degree(self, D):
        # get D^-1/2
        D = np.linalg.inv(D)
        D = np.power(D, .5)

        return D

    def __get_eta(self, batch_size):
        eta = torch.zeros((batch_size, 2), dtype=torch.float)
        for i in range(batch_size):
            alpha = np.random.rand()

            if alpha < .25:
                eta[i, 1] = 1
            else:
                eta[i, 0] = 1

        return eta

    def __get_gamma(self, batch_size):
        gamma = torch.zeros((batch_size, 2), dtype=torch.float)

        for i in range(batch_size):
            beta = np.random.rand()

            if beta < .33:
                gamma[i, 1] = 1
            else:
                gamma[i, 0] = 1

        return gamma

    def __gen_neg_samples_discriminator(self):
        pass

    def __gen_pos_samples_gcn(self, regions, idx_map, h_graph, batch_size):
        pos_sample_map = dict()
        n_h_dims = h_graph.shape[1]
        pos_samples = torch.zeros((batch_size, n_h_dims), dtype=torch.float)

        for id, mtx_idx in idx_map.items():
            region_i = regions[id]
            pos_sample = np.random.choice(list(region_i.adjacent.keys()))
            pos_sample_map[mtx_idx] = idx_map[pos_sample]
            pos_samples[mtx_idx, :] = h_graph[idx_map[pos_sample], :]

        return pos_samples

    def __gen_neg_samples_gcn(self, n_neg_samples, adj_mtx, h_graph, idx_map, batch_size):


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



    def run_train_job(self,region_grid, epochs, lr, n_neg_samples=15):

        optimizer = self.get_optimizer(lr=lr)

        region_mtx_map = region_grid.matrix_idx_map

        A = region_grid.adj_matrix
        D = region_grid.degree_matrix
        W = region_grid.weighted_mtx
        X = region_grid.feature_matrix
        # preprocess step for graph matrices
        A_hat = self.__preprocess_adj(A)
        D_hat = self.__preprocess_degree(D)

        # Cast matrices to torch.tensor
        A_hat = torch.from_numpy(A_hat).type(torch.FloatTensor)
        D_hat = torch.from_numpy(D_hat).type(torch.FloatTensor)
        W = torch.from_numpy(W).type(torch.FloatTensor)
        X = torch.from_numpy(X).type(torch.FloatTensor)

        img_tensor = torch.Tensor(region_grid.img_tensor)

        batch_size = A.shape[0]


        # Ground truth for discriminator
        eta = self.__get_eta(batch_size)
        # ground truth for spatial reconstruction
        gamma = self.__get_gamma(batch_size)

        print("Beginning training job: epochs: {}, batch size: {}".format(epochs, batch_size))
        for i in range(epochs):

            optimizer.zero_grad()
            # forward + backward + optimize
            logits, h_global, image_hat, graph_proximity, h_graph, h_image = mod.forward(X=X, A=A_hat, D=D_hat,
                                                                                         img_tensor=img_tensor)
            # generate positive samples for gcn
            gcn_pos_samples = self.__gen_pos_samples_gcn(region_grid.regions, region_mtx_map, h_graph, batch_size)
            # generate negative samples for gcn
            gcn_neg_samples = self.__gen_neg_samples_gcn(n_neg_samples, A, h_graph, region_mtx_map, batch_size)

            # Get different objectives
            L_graph = self.loss_graph(h_graph, gcn_pos_samples, gcn_neg_samples)
            L_edge_weights = self.loss_weighted_edges(graph_proximity, W)
            L_disc = self.loss_disc(eta, logits)
            L_ae = self.loss_ae(img_tensor, image_hat)

            loss = self.loss_function(L_graph, L_edge_weights, L_disc, L_ae)
            loss.backward()
            optimizer.step()
            # print statistics
            # loss.item()
            print("Epoch: {}, Train Loss {:.4f}".format(i+1, loss.item()))




if __name__ == "__main__":
    c = get_config()
    file = open(c["poi_file"], 'rb')
    img_dir = c['path_to_image_dir']
    region_grid = RegionGrid(50, poi_file=file, img_dir=img_dir, w_mtx_file=c['flow_mtx_file'], load_imgs=False)

    mod = RegionEncoder(n_nodes=2500, n_nodal_features=552, h_dim_graph=64, lambda_ae=.1, lambda_edge=.1, lambda_g=.1)
    mod.run_train_job(region_grid, epochs=100, lr=.05)

