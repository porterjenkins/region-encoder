import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from model.get_karate_data import *


class GraphConv(nn.Module):

    def __init__(self, input_dim, output_dim):
        super(GraphConv, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.weight_tensor = torch.Tensor(input_dim, output_dim)

    def forward(self, input, adj):
        # G_0 = torch.mm(A, D)  # no normalization - data should be preprocessed
        support = torch.mm(input, self.weight_tensor)
        output = torch.mm(adj, support)
        return output


class GCN(nn.Module):
    """
    Graph ConvNet Model (via Kipf and Welling ICLR'2017)
    Three matrices required:
        - X: Nodal feature matrix that has been normalized
        - A: Adjacency matrix (A + I)
    """

    def __init__(self, n_features, n_classes, h_dim_size=16):
        super(GCN, self).__init__()
        self.n_features = n_features
        self.n_classes = n_features
        self.n_hidden = h_dim_size

        self.gc_0 = GraphConv(n_feature, h_dim_size)
        self.gc_1 = GraphConv(h_dim_size, n_classes)

        # fully connected layer 1
        self.fcl_0 = nn.Linear(n_features, 512, bias=True)
        # Output layer for link prediction
        self.fcl_1 = nn.Linear(512, h_dim_size, bias=True)

    def forward(self, x, a):
        x = self.gc_0(x, a)  # graph 1
        x = self.fcl_0(x)  # linear layer 1 - not in tensor impl
        x = F.relu(x)
        x = self.gc_1(x, a)  # graph 2
        x = self.fcl_1(x)  # output layer

        return x

    def get_optimizer(self):
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(self.parameters(), lr=0.1, momentum=0.9)

        return criterion, optimizer

    def run_train_job(self, X_train, y_train, A, n_epoch):
        cross_entropy, optimizer = self.get_optimizer()

        for epoch in range(n_epoch):  # loop over the dataset multiple times

            running_loss = 0.0
            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            output, H = gcn.forward(X_train, A)
            loss = cross_entropy(output, y_train)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            print("Epoch: {}, Train Loss {:.4f}".format(epoch, loss))

        print('Finished Training')


def normalize_adjacency_matrix(adj):
    row_sum = np.array(adj.sum(1))  # D
    d_inv_sqrt = np.power(row_sum, -0.5).flatten()  # D -1/2
    # set infinities to 0
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat = np.diag(d_inv_sqrt)  # diagonal node degree

    # i think
    # D^−1/2A^D^−1/2
    return adj.dot(d_mat).transpose().dot(d_mat)


def preprocess(adj):
    # Ahat = A + I
    return normalize_adjacency_matrix(adj + np.eye(adj.shape[0]))


if __name__ == "__main__":
    A = preprocess(get_adj_mtx())
    X = get_features()
    W = get_weighted_graph(A.shape[0])

    n_feature = X.shape[1]

    y = get_labels(n_samples=X.shape[0], class_probs=[.2, .5, .2, .1])

    A = torch.from_numpy(A).type(torch.FloatTensor)
    X = torch.from_numpy(X).type(torch.FloatTensor)
    labels = torch.from_numpy(y).type(torch.LongTensor)

    gcn = GCN(n_features=2, n_classes=4, h_dim_size=16)

    gcn.run_train_job(X_train=X, y_train=labels, A=A, n_epoch=250)
