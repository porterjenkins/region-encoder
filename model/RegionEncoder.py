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

class LinearLayer_1(nn.Module):
    def __init__(self, n_features, out_features):
        super(LinearLayer_1, self).__init__()
        self.lin_layer = nn.Linear(n_features, out_features, bias=True)

    def forward(self, x):
        h = F.relu(self.lin_layer(x))
        return h


class LinearLayer_2(nn.Module):
    def __init__(self, n_features, out_features):
        super(LinearLayer_2, self).__init__()
        self.lin_layer = nn.Linear(n_features, out_features, bias=True)

    def forward(self, h):
        y_hat = torch.sigmoid(self.lin_layer(h))
        return y_hat



class RegionEncoderTest(nn.Module):
    """
    Implementatino of proposed model for
    Multi-Modal Region Encoding (MMRE)
    """
    def __init__(self):
        super(RegionEncoderTest, self).__init__()
        self.l_1 = LinearLayer_1(8, 4)
        self.l_2 = LinearLayer_2(4, 1)

    def forward(self, X):
        h = self.l_1.forward(X)
        y_hat = self.l_2.forward(h)

        return y_hat

class RegionEncoder(nn.Module):
    """
    Implementatino of proposed model for
    Multi-Modal Region Encoding (MMRE)
    """
    def __init__(self, n_nodes, n_nodal_features, h_dim_graph=4, h_dim_img=32, h_dim_disc=32):
        super(RegionEncoder, self).__init__()
        self.graph_conv_net = GCN(n_nodes=n_nodes, n_features=n_nodal_features, h_dim_size=h_dim_graph)
        self.auto_encoder = AutoEncoder(h_dim_size=h_dim_img)
        self.discriminator = DiscriminatorMLP(x_features=h_dim_graph, z_features=h_dim_img, h_dim_size=h_dim_disc)

    def forward(self, X, A, D, img_tensor):
        #h = self.l_1.forward(X)
        #y_hat = self.l_2.forward(h)
        h_graph = self.graph_conv_net.forward(X, A, D)
        image_hat, h_image = self.auto_encoder.forward(img_tensor)
        y_hat, global_embedding = self.discriminator.forward(x=h_graph, z=h_image)

        return y_hat, global_embedding

if __name__ == "__main__":
    batch_size = 34

    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    trainset = torchvision.datasets.CIFAR10(root='../tutorials/data', train=True,
                                            download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                              shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(root='../tutorials/data', train=False,
                                           download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                             shuffle=False, num_workers=2)

    A = get_adj_mtx()
    X = get_features()

    n_nodes = X.shape[0]
    n_feature = X.shape[1]

    D_hat = get_degree_mtx(A)
    A_hat = get_a_hat(A)

    y = get_labels(n_samples=X.shape[0], class_probs=[.2, .5, .2, .1])

    A = torch.from_numpy(A).type(torch.FloatTensor)
    X = torch.from_numpy(X).type(torch.FloatTensor)
    D_hat = torch.from_numpy(D_hat).type(torch.FloatTensor)
    A_hat = torch.from_numpy(A_hat).type(torch.FloatTensor)

    dataiter = iter(trainloader)
    images, labels = dataiter.next()

    mod = RegionEncoder(n_nodes=batch_size, n_nodal_features=2)

    optimizer = optim.SGD(mod.parameters(), lr=0.1, momentum=0.9)
    cross_entropy = torch.nn.BCELoss()
    n_epoch = 100

    optimizer.zero_grad()

    # forward + backward + optimize
    #y_hat = mod.forward(X_train)
    y_hat, h = mod.forward(X=X, A=A_hat, D=D_hat, img_tensor=images)
    print(y_hat)
    print(h)
    #loss = cross_entropy(y_hat, y_train)
    #loss.backward()
    #optimizer.step()

    # print statistics
    #loss.item()
    #print("Epoch: {}, Train Loss {:.4f}".format(i, loss.item()))