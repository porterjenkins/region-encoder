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
                 lambda_ae=.1, lambda_g=.1, lambda_edge=.1, lambda_weight_decay=.01, img_dims=(640,640),
                 neg_samples_disc=None, neg_samples_gcn = 10):
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

        # Sampling parameters
        if neg_samples_disc is None:
            self.neg_samples_disc = n_nodes
        else:
            self.neg_samples_disc = neg_samples_disc
        self.neg_samples_gcn = neg_samples_gcn
        self.n_nodes = n_nodes

        # Final model hidden state
        self.embedding = torch.Tensor

    def forward(self, X, A, D, img_tensor):

        # Forward step for graph data
        h_graph = self.graph_conv_net.forward(X, A, D)
        graph_proximity = self._get_weighted_proximity(h_graph)

        # Forward step for image data
        image_hat, h_image = self.auto_encoder.forward(img_tensor)

        # generate negative samples for discriminator
        h_graph_neg, h_img_neg = self.__gen_neg_samples_disc(h_graph, h_image)

        # concat positive and negatives samples for discriminator
        h_graph_cat = torch.cat([h_graph, h_graph_neg], dim=0)
        h_img_cat = torch.cat([h_image, h_img_neg], dim=0)

        # forward step for discriminator (all data)
        logits, h_global = self.discriminator.forward(x=h_graph_cat, z=h_img_cat, activation=False)

        return logits, h_global, image_hat, graph_proximity, h_graph, h_image, h_graph_neg, h_img_neg


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

    def __gen_eta(self, pos_tens, neg_tens):

        eta = torch.zeros(pos_tens.shape[0] + neg_tens.shape[0], 2)
        # Positive labels
        eta[:pos_tens.shape[0], 1] = 1
        # negative labels
        eta[-neg_tens.shape[0]:, 0] = 1


        return eta

    def __gen_neg_samples_disc(self, h_graph, h_image):
        idx = np.arange(self.n_nodes)

        neg_idx_graph = np.random.choice(idx, size=self.neg_samples_disc, replace=True)
        neg_idx_image = np.random.choice(idx, size=self.neg_samples_disc, replace=True)

        graph_equal_img = np.where(neg_idx_graph == neg_idx_image)[0]

        # If (due to random chance) negative sampled graph and image representations are from same region,
        # resample until they are true negative samples
        for i in graph_equal_img:
            while neg_idx_graph[i] == neg_idx_image[i]:
                new_idx = np.random.randint(0, self.n_nodes)
                neg_idx_graph[i] = new_idx

        h_graph_neg = h_graph[neg_idx_graph,:]
        h_img_neg = h_image[neg_idx_image, :]


        return h_graph_neg, h_img_neg

    def write_embeddings(self, fname):

        arr = self.embedding.data.numpy()
        h_dim = arr.shape[1]

        with open(fname, 'w') as f:
            f.write("{} {} \n".format(self.n_nodes, h_dim))

            #for cntr, embedding_vector in enumerate(arr):
            for region_idx in range(self.n_nodes):
                embedding_vector = arr[region_idx, :]
                f.write("{} ".format(region_idx))

                cntr = 0
                for element in embedding_vector:
                    if cntr == (h_dim-1):
                        f.write("{}".format(element))
                    else:
                        f.write("{} ".format(element))

                    cntr += 1


                f.write("\n")

    def __earling_stop(self, seq, tol, order):
        """
        Determine early stopping of training job
        :param seq:
        :param tol:
        :param order:
        :return:
        """
        diffs = np.abs(np.diff(seq[-order:]))
        diffs_bool = diffs < tol
        diff_cnt_true = np.sum(diffs_bool)

        if len(seq) > order and diff_cnt_true == order:
            return True
        else:
            return False


    def run_train_job(self,region_grid, epochs, lr, tol=.001, tol_order=5):

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
        print("Beginning training job: epochs: {}, batch size: {}".format(epochs, batch_size))

        loss_seq = list()
        for i in range(epochs):

            optimizer.zero_grad()
            # forward + backward + optimize
            logits, h_global, image_hat, graph_proximity, h_graph, h_image, h_graph_neg,\
            h_image_neg = mod.forward(X=X, A=A_hat, D=D_hat, img_tensor=img_tensor)

            # generate positive samples for gcn
            gcn_pos_samples = self.__gen_pos_samples_gcn(region_grid.regions, region_mtx_map, h_graph, batch_size)
            # generate negative samples for gcn
            gcn_neg_samples = self.__gen_neg_samples_gcn(self.neg_samples_gcn, A, h_graph, region_mtx_map, batch_size)
            # get labels for discriminator
            eta = self.__gen_eta(pos_tens=h_graph, neg_tens=h_graph_neg)
            # Add noise to images
            img_noisey = AutoEncoder.add_noise(img_tensor, noise_factor=.25)

            # Get different objectives
            L_graph = self.loss_graph(h_graph, gcn_pos_samples, gcn_neg_samples)
            L_edge_weights = self.loss_weighted_edges(graph_proximity, W)
            L_disc = self.loss_disc(eta, logits)
            L_ae = self.loss_ae(img_noisey, image_hat)

            loss = self.loss_function(L_graph, L_edge_weights, L_disc, L_ae)
            loss.backward()
            optimizer.step()
            loss_seq.append(loss.item())

            if np.isnan(loss_seq[-1]):
                print("Exploding/Vanishing gradient: loss = nan")
                break
            elif self.__earling_stop(loss_seq, tol, tol_order):
                print("Terminating early: Epoch: {}, Train Loss {:.4f} (gcn: {:.4f}, edge: {:.4f}, discriminator: {:.4f}"
                      " autoencoder: {:.4f})".format(i+1, loss_seq[-1], L_graph, L_edge_weights, L_disc, L_ae))
                break
            else:
                print("Epoch: {}, Train Loss {:.4f} (gcn: {:.4f}, edge: {:.4f}, discriminator: {:.4f}"
                      " autoencoder: {:.4f})".format(i + 1, loss_seq[-1], L_graph, L_edge_weights, L_disc, L_ae))

        self.embedding = h_global





if __name__ == "__main__":
    c = get_config()
    region_grid = RegionGrid(config=c, load_imgs=True)
    n_nodes = len(region_grid.regions)

    mod = RegionEncoder(n_nodes=n_nodes, n_nodal_features=552, h_dim_graph=64, lambda_ae=.1, lambda_edge=.1, lambda_g=0.1, neg_samples_gcn=10)
    mod.run_train_job(region_grid, epochs=100, lr=.005, tol_order=3)
    mod.write_embeddings(c['embedding_file'])


