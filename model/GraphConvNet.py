import torch
import torch.nn as nn
import torch.nn.functional as F
from get_karate_data import *


class GCN(nn.Module):
    def __init__(self, n_nodes, n_features):
        super(GCN, self).__init__()
        self.n_nodes = n_nodes
        self.n_features = n_feature
        # fully connected layer 1
        self.fcl_0 = nn.Linear(n_features, 8, bias=True)
        # fully connected layer 2
        self.fcl_1 = nn.Linear(8, 8, bias=True)



    def forward(self, X, A, D):
        # G = D*A*D*X*W
        G_0 = torch.mm(torch.mm(A, torch.mm(A,D)), X)
        H_0 = F.relu(self.fcl_0(G_0))

        G_1 = torch.mm(torch.mm(A, torch.mm(A,D)), H_0)
        H_1 = self.fcl_1(G_1)

        return H_1



if __name__ == "__main__":

    A = get_adj_mtx()
    X = get_features()

    n_nodes = X.shape[0]
    n_feature = X.shape[1]

    D = get_degree_mtx(A)
    A_hat = get_a_hat(A)


    A = torch.from_numpy(A).type(torch.FloatTensor)
    X = torch.from_numpy(X).type(torch.FloatTensor)
    D = torch.from_numpy(D).type(torch.FloatTensor)
    A_hat = torch.from_numpy(A_hat).type(torch.FloatTensor)

    gcn = GCN(n_nodes=n_nodes, n_features=2)

    print(gcn.forward(X, A_hat, D))

