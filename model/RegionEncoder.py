import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import torch
import torch.nn as nn
import torch.optim as optim
from model.AutoEncoder import AutoEncoder
from model.GraphConvNet import GCN
from model.discriminator import DiscriminatorMLP
from config import get_config
import numpy as np
import matplotlib.pyplot as plt
from grid.create_grid import RegionGrid
from model.utils import write_embeddings


class RegionEncoder(nn.Module):
    """
    Implementation of proposed model for
    Multi-Modal Region Encoding (MMRE)
    """

    def __init__(self, n_nodes, n_nodal_features, h_dim_graph=32, h_dim_img=32, h_dim_size=32,
                 lambda_ae=.1, lambda_g=.1, lambda_edge=.1, lambda_weight_decay=1e-4, img_dims=(200, 200),
                 neg_samples_disc=None, neg_samples_gcn=10, context_gcn=4):
        super(RegionEncoder, self).__init__()

        # Ensure consistent hidden size for discriminator
        #assert(h_dim_img == h_dim_graph)
        #assert(h_dim_size == h_dim_graph)
        # Model Layers
        self.graph_conv_net = GCN(n_features=n_nodal_features, h_dim_size=h_dim_graph)
        self.auto_encoder = AutoEncoder(h_dim_size=h_dim_img, img_dims=img_dims)
        self.discriminator = DiscriminatorMLP(x_features=h_dim_graph, z_features=h_dim_img, h_dim_size=h_dim_size)

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
        self.context_gcn = context_gcn
        self.n_nodes = n_nodes
        self.img_dims = img_dims
        self.h_dim_img = h_dim_img
        self.h_dim_graph = h_dim_graph


        # Final model hidden state
        self.embedding = torch.Tensor
        # Store loss values
        self.loss_seq = []
        self.loss_seq_gcn = []
        self.loss_seq_edge = []
        self.loss_seq_disc = []
        self.loss_seq_ae = []
        self.use_cuda = torch.cuda.is_available()

    def forward(self, X, H_img, H_graph, img_batch=None, update_img_weights=False, batch_idx=None):
        """

        :param X:
        :param img_batch:
        :param H_img:
        :param update_img_weight:
        :return:
        """

        if update_img_weights:
            # Forward step for image data
            image_hat, h_image_step = self.auto_encoder.forward(img_batch)
            # Update Hidden State for images
            H_img[batch_idx, :] = h_image_step
        else:
            # Forward step for graph data
            h_graph_step = self.graph_conv_net.forward(X)
            # Update Hidden State for graph
            H_graph = h_graph_step
            #image_hat, h_image_step = self.auto_encoder.forward(H_img, decode_only=True)
            image_hat = None

        graph_proximity = GCN.get_weighted_proximity(H_graph)

        # generate negative samples for discriminator
        h_graph_neg, h_img_neg = self.__gen_neg_samples_disc(H_graph, H_img)

        # concat positive and negatives samples for discriminator
        h_graph_cat = torch.cat([H_graph, h_graph_neg], dim=0)
        h_img_cat = torch.cat([H_img, h_img_neg], dim=0)

        # forward step for discriminator (all data)
        logits, h_global = self.discriminator.forward(x=h_graph_cat, z=h_img_cat, activation=False)

        return logits, h_global, graph_proximity, H_graph, h_graph_neg, h_img_neg, H_img, image_hat

    def weight_decay(self):
        reg = 0

        for p in self.parameters():
            layer = p.data
            reg += self.lambda_wd * torch.norm(layer)

        return reg

    def get_optimizer(self, lr):
        optimizer = optim.SGD(self.parameters(), lr=lr, momentum=0.9)
        #optimizer = optim.Adam(self.parameters(), lr=lr)

        return optimizer

    def loss_disc(self, eta, eta_logits):
        loss_disc = self.bce_logits(eta_logits, eta)
        return loss_disc

    def loss_function(self, L_graph, L_edge_weights, L_disc, L_ae, reg):
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
        :param reg: Regularization term
        :return:
        """

        L = L_disc + self.lambda_ae * L_ae + self.lambda_g * L_graph + self.lambda_edge * L_edge_weights + reg

        return L

    def __gen_eta(self, pos_tens, neg_tens):

        eta = torch.zeros(pos_tens.shape[0] + neg_tens.shape[0], 2)
        # Positive labels
        eta[:pos_tens.shape[0], 1] = 1
        # negative labels
        eta[-neg_tens.shape[0]:, 0] = 1
        if self.use_cuda:
            eta = eta.cuda()
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

        h_graph_neg = h_graph[neg_idx_graph, :]
        h_img_neg = h_image[neg_idx_image, :]

        return h_graph_neg, h_img_neg

    def plt_learning_curve(self, fname=None, log_scale=True, plt_all=True):
        x = np.arange(1, len(self.loss_seq) + 1)

        if log_scale:

            plt.plot(x, np.log(self.loss_seq), label='Total Loss')
            if plt_all:
                plt.plot(x, np.log(self.loss_seq_gcn), label="SkipGram GCN")
                plt.plot(x, np.log(self.loss_seq_edge), label='Weighted Edge')
                plt.plot(x, np.log(self.loss_seq_ae), label="AutoEncoder")
                plt.plot(x, np.log(self.loss_seq_disc), label='Discriminator')

            plt.xlabel("Epochs")
            plt.ylabel("Loss (log scale)")

        else:

            plt.plot(x, self.loss_seq, label='Total Loss')
            if plt_all:
                plt.plot(x, self.loss_seq_gcn, label="SkipGram GCN")
                plt.plot(x, self.loss_seq_edge, label='Weighted Edge')
                plt.plot(x, self.loss_seq_ae, label="AutoEncoder")
                plt.plot(x, self.loss_seq_disc, label='Discriminator')

            plt.xlabel("Epochs")
            plt.ylabel("Loss")

        plt.legend(loc='best')

        if fname is not None:
            dir = os.path.dirname(fname)
            if not os.path.exists(dir):
                os.makedirs(dir)

            plt.savefig(fname)
            plt.clf()
            plt.close()

        else:
            plt.show()

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

    def run_train_job(self, region_grid, epochs, lr, batch_size, tol=.001, tol_order=5, n_graph_updates=2):
        optimizer = self.get_optimizer(lr=lr)

        region_mtx_map = region_grid.matrix_idx_map

        A = region_grid.adj_matrix
        D = region_grid.degree_matrix
        W = region_grid.weighted_mtx
        X = region_grid.feature_matrix
        # preprocess step for graph matrices
        A_hat = GCN.preprocess_adj(A)
        D_hat = GCN.preprocess_degree(D)

        region_grid.arr_size(region_grid.img_tensor)

        # Cast matrices to torch.tensor
        A_hat = torch.from_numpy(A_hat).type(torch.FloatTensor)
        D_hat = torch.from_numpy(D_hat).type(torch.FloatTensor)
        W = torch.from_numpy(W).type(torch.FloatTensor)
        X = torch.from_numpy(X).type(torch.FloatTensor)

        region_grid.img_tensor = torch.from_numpy(region_grid.img_tensor).type(torch.FloatTensor)

        n_samples = A.shape[0]

        I_batch = torch.zeros((batch_size, 3, self.img_dims[1], self.img_dims[1]))
        H_img = torch.rand(n_samples, self.h_dim_img)
        H_graph = torch.rand(n_samples, self.h_dim_img)

        print("Starting training job: epochs: {}, batch size: {}, learning rate:{}".format(epochs, batch_size, lr))

        self.graph_conv_net.adj = torch.mm(D_hat, torch.mm(A_hat, D_hat))
        if self.use_cuda:
            self.graph_conv_net.adj = self.graph_conv_net.adj.cuda()
            X = X.cuda()
            W = W.cuda()
            I_batch = I_batch.cuda()
            H_img = H_img.cuda()

        for i in range(epochs):

            print("Epoch: {}".format(i + 1))
            print('Updating Graph Weights:')
            for graph_step in range(n_graph_updates):
                if self.use_cuda:
                    print("CUDA Memory: {}".format(torch.cuda.memory_allocated()))
                optimizer.zero_grad()
                # FIX: Weight_img - UPDATE: Weights_graph, Weights_disc
                logits, h_global, graph_proximity, H_graph, h_graph_neg, h_image_neg, _, _ = self.forward(X, H_img, H_graph)

                # Generate context (positive samples) and negative samples for SkipGram Loss
                gcn_pos_samples, gcn_neg_samples, neg_probs = GCN.gen_skip_gram_samples(self.context_gcn, self.neg_samples_gcn,
                                                                                        H_graph, n_samples, region_mtx_map,
                                                                                        region_grid.regions, A, D)

                # get labels for discriminator
                eta = self.__gen_eta(pos_tens=H_graph, neg_tens=h_graph_neg)
                # Get different objectives
                L_graph = GCN.skip_gram_loss(H_graph, gcn_pos_samples, gcn_neg_samples, neg_probs)
                emp_proximity = W / torch.sum(W)
                L_edge_weights = GCN.loss_weighted_edges(graph_proximity, emp_proximity)
                L_disc = self.loss_disc(eta, logits)
                #L_ae = AutoEncoder.loss_mse(image_hat, I_batch)
                L_ae = torch.tensor(0.0)

                # regularize weights
                weight_decay = self.weight_decay()
                loss = self.loss_function(L_graph, L_edge_weights, L_disc, L_ae, weight_decay)
                loss.backward(retain_graph=True, create_graph=False)
                optimizer.step()


                print("-->Graph Step: {}: Train Loss {:.4f} (gcn: {:.4f}, edge: {:.4f}, discriminator: {:.4f}"
                      " autoencoder: {:.4f})".format(graph_step + 1, loss.item(), L_graph, L_edge_weights, L_disc, L_ae))

                # FIX: Weights_graph -- UPDATE: Weights_img, Weights_disc
                permute_idx = np.random.permutation(np.arange(n_samples))


            print("Updating Image Weights:")
            n_steps = int(n_samples / batch_size)
            for step in range(n_steps):

                # zero the parameter gradients
                optimizer.zero_grad()

                start_idx = step * batch_size
                end_idx = start_idx + batch_size
                batch_idx = permute_idx[start_idx:end_idx]

                I_batch[:, :, :, :] = region_grid.img_tensor[batch_idx, :, :, :]
                # Add noise to images
                img_noisey = AutoEncoder.add_noise(I_batch, noise_factor=.25, cuda=self.use_cuda)

                logits, h_global, graph_proximity, H_graph, h_graph_neg, h_image_neg, H_img, image_hat = \
                    self.forward(X, H_img, H_graph, img_batch=img_noisey, update_img_weights=True, batch_idx=batch_idx)

                L_ae = AutoEncoder.loss_mse(image_hat, I_batch)


                # get labels for discriminator
                eta = self.__gen_eta(pos_tens=H_graph, neg_tens=h_graph_neg)
                L_graph = torch.tensor(0.0)
                L_edge_weights = torch.tensor(0.0)
                L_disc = self.loss_disc(eta, logits)

                # regularize weights
                weight_decay = self.weight_decay()
                loss = self.loss_function(L_graph, L_edge_weights, L_disc, L_ae, weight_decay)
                #if step == (n_steps - 1):
                #    loss.backward(retain_graph=False)
                #else:

                loss.backward(retain_graph=True)
                optimizer.step()


                print("--> Image Step {}: Train Loss {:.4f} (gcn: {:.4f}, edge: {:.4f}, discriminator: {:.4f}"
                      " autoencoder: {:.4f})".format(step+1, loss.item(), L_graph, L_edge_weights, L_disc, L_ae))

                if self.use_cuda:
                    torch.cuda.empty_cache()


            del loss

            # store loss values for learning curve
            self.loss_seq.append(loss.item())
            self.loss_seq_gcn.append(self.lambda_g * L_graph.item())
            self.loss_seq_edge.append(self.lambda_edge * L_edge_weights.item())
            self.loss_seq_disc.append(L_disc.item())
            self.loss_seq_ae.append(self.lambda_ae * L_ae.item())

            if np.isnan(self.loss_seq[-1]):
                print("Exploding/Vanishing gradient: loss = nan")
                break
            elif self.__earling_stop(self.loss_seq, tol, tol_order):
                print(
                    "Terminating early: Epoch: {}, Train Loss {:.4f} (gcn: {:.4f}, edge: {:.4f}, discriminator: {:.4f}"
                    " autoencoder: {:.4f})".format(i + 1, self.loss_seq[-1], L_graph, L_edge_weights, L_disc, L_ae))
                break

        self.embedding = h_global


if __name__ == "__main__":

    c = get_config()
    region_grid = RegionGrid(config=c)
    region_grid.load_img_data(std_img=True)
    region_grid.load_weighted_mtx()
    n_nodes = len(region_grid.regions)

    # hyperparameters
    n_nodal_features = region_grid.feature_matrix.shape[1]
    h_dim_graph = 64
    h_dim_img = 32
    h_dim_size = int(c['hidden_dim_size'])
    lambda_ae = .5
    lambda_edge = .1
    lambda_g = .1
    context_gcn = 4
    neg_samples_gcn = 10
    epochs = 50
    learning_rate = .1
    neg_samples_disc = 10
    batch_size = 25


    if len(sys.argv) > 1:
        epochs = int(sys.argv[1])
        learning_rate = float(sys.argv[2])
        batch_size = int(sys.argv[3])

    mod = RegionEncoder(n_nodes=n_nodes,
                        n_nodal_features=n_nodal_features,
                        h_dim_graph=h_dim_graph,
                        h_dim_img=h_dim_img,
                        lambda_ae=lambda_ae,
                        lambda_edge=lambda_edge,
                        lambda_g=lambda_g,
                        neg_samples_gcn=neg_samples_gcn,
                        h_dim_size=h_dim_size,
                        context_gcn=context_gcn,
                        neg_samples_disc=neg_samples_disc)
    mod.run_train_job(region_grid, epochs=epochs, lr=learning_rate, tol_order=3, batch_size=batch_size,
                      n_graph_updates=1)

    if torch.cuda.is_available():
        embedding = mod.embedding.data.cpu().numpy()
    else:
        embedding = mod.embedding.data.numpy()

    write_embeddings(arr=embedding, n_nodes=n_nodes, fname=c['embedding_file'])
    mod.plt_learning_curve("plots/region-learning-curve.pdf", plt_all=False, log_scale=False)
