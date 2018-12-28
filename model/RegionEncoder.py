import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch.optim as optim
from model.AutoEncoder import AutoEncoder
from model.GraphConvNet import GCN
from model.discriminator import DiscriminatorMLP
import torchvision
import torchvision.transforms as transforms
from model.get_karate_data import *


class RegionEncoder(nn.Module):
    """
    Implementation of proposed model for
    Multi-Modal Region Encoding (MMRE)
    """
    def __init__(self, n_nodes, n_nodal_features, h_dim_graph=4, h_dim_img=32, h_dim_disc=32,
                 lambda_ae=.1, lambda_g=.1, lambda_edge=.1, lambda_weight_decay=.01):
        super(RegionEncoder, self).__init__()
        # Model Layers
        self.graph_conv_net = GCN(n_nodes=n_nodes, n_features=n_nodal_features, h_dim_size=h_dim_graph, n_classes=4)
        self.auto_encoder = AutoEncoder(h_dim_size=h_dim_img)
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
        graph_logits, h_graph = self.graph_conv_net.forward(X, A, D)
        graph_proximity = self._get_weighted_proximity(h_graph)

        # Forward step for image data
        image_hat, h_image = self.auto_encoder.forward(img_tensor)

        # forward step for discriminator (all data)
        logits, h_global = self.discriminator.forward(x=h_graph, z=h_image, activation=False)

        return logits, h_global, image_hat, graph_proximity, h_graph, graph_logits, h_image


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

    def loss_graph(self, graph_logits, gamma):
        """

        :param graph_logits:
        :param gamma:
        :return:
        """
        loss_g = self.bce_logits(graph_logits, gamma)
        return loss_g

    def loss_disc(self, eta, eta_logits):
        loss_disc = self.bce_logits(eta_logits, eta)
        return loss_disc

    def loss_gcn(self, h_graph, graph_label):
        loss_graph = self.cross_entropy(h_graph, graph_label)

        return loss_graph

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

    def loss_function(self, eta, eta_logits,  img_input, img_reconstruction,
                      h_graph, graph_logits, gamma, learned_graph_prox, emp_graph_prox):
        """
        Global loss function for model. Loss has the following components:
            - Reconstruction of spatial graph
            - Prediction of flow graph
            - Reconstruction of image
            - Error of the discriminator

        :param eta:
        :param eta_logits:
        :param img_input:
        :param img_reconstruction:
        :param graph_label_pred:
        :param graph_label:
        :return:
        """
        #L_gcn = self.loss_gcn(h_graph, graph_label)
        L_graph = self.loss_graph(graph_logits, gamma)
        L_edge_weights = self.loss_weighted_edges(learned_graph_prox, emp_graph_prox)
        L_disc = self.loss_disc(eta, eta_logits)
        L_ae = self.loss_ae(img_input, img_reconstruction)


        #regularizer = self.weight_decay()

        L = L_disc + self.lambda_ae * L_ae + self.lambda_g * L_graph + self.lambda_edge * L_edge_weights



        return L


    def run_train_job(self, epochs, lr):
        optimizer = self.get_optimizer(lr=lr)
        batch_size = 34

        transform = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

        trainset = torchvision.datasets.CIFAR10(root='../tutorials/data', train=True,
                                                download=True, transform=transform)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                                  shuffle=True, num_workers=2)

        #testset = torchvision.datasets.CIFAR10(root='../tutorials/data', train=False,
        #                                       download=True, transform=transform)
        #testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
        #                                         shuffle=False, num_workers=2)

        A = get_adj_mtx()
        X = get_features()
        W = get_weighted_graph(A.shape[0])

        D_hat = get_degree_mtx(A)
        A_hat = get_a_hat(A)

        graph_label = get_labels(n_samples=X.shape[0], class_probs=[.2, .5, .2, .1])

        graph_label = torch.from_numpy(graph_label).type(torch.LongTensor)

        X = torch.from_numpy(X).type(torch.FloatTensor)
        W = torch.from_numpy(W).type(torch.FloatTensor)

        D_hat = torch.from_numpy(D_hat).type(torch.FloatTensor)
        A_hat = torch.from_numpy(A_hat).type(torch.FloatTensor)

        dataiter = iter(trainloader)
        images, labels = dataiter.next()

        eta = torch.zeros((batch_size, 2), dtype=torch.float)
        gamma = torch.zeros((batch_size, 2), dtype=torch.float)

        for i in range(batch_size):
            alpha = np.random.rand()
            beta = np.random.rand()

            if alpha < .25:
                eta[i, 1] = 1
            else:
                eta[i, 0] = 1


            if beta < .33:
                gamma[i, 1] = 1
            else:
                gamma[i, 0] = 1



        for i in range(epochs):

            optimizer.zero_grad()
            # forward + backward + optimize
            logits, h_global, image_hat, graph_proximity, h_graph, graph_logits, h_image = mod.forward(X=X, A=A_hat,
                                                                                                       D=D_hat,
                                                                                                       img_tensor=images)


            loss = mod.loss_function(eta=eta, eta_logits=logits, img_input=images, img_reconstruction=image_hat,
                                     h_graph=h_graph, graph_logits=graph_logits, gamma=gamma,
                                     learned_graph_prox=graph_proximity,
                                     emp_graph_prox=W)

            # loss = cross_entropy(y_hat, y_train)
            loss.backward()
            optimizer.step()
            # print statistics
            # loss.item()
            print("Epoch: {}, Train Loss {:.4f}".format(i+1, loss.item()))



if __name__ == "__main__":

    mod = RegionEncoder(n_nodes=34, n_nodal_features=2, lambda_ae=.1, lambda_edge=.1, lambda_g=.1)
    mod.run_train_job(epochs=250, lr=.05)

