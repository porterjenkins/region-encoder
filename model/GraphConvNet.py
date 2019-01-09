import torch
import torch.nn as nn
import torch.nn.functional as F

from model.get_karate_data import *


class GCN(nn.Module):
    """
    Graph ConvNet Model (via Kipf and Welling ICLR'2017)
    Two matrices required:
        - X: Nodal feature matrix that has been normalized
        - A: Adjacency matrix
    """

    def __init__(self, adj, n_features, n_classes, n_hidden=16):
        super(GCN, self).__init__()
        self.n_features = n_features
        self.n_classes = n_features
        self.n_hidden = n_hidden
        # pre process adj matrix ( D * (A + I) * D)
        self.adj = self.normalize_adj(adj)

        # fully connected layer 1
        self.fcl_0 = nn.Linear(n_features, 512, bias=True)
        # Output layer for link prediction
        self.fcl_1 = nn.Linear(512, n_hidden, bias=True)

        if torch.cuda.is_available():
            self.adj = self.adj.cuda()
            self.fcl_0 = self.fcl_0.cuda()
            self.fcl_1 = self.fcl_1.cuda()

    def forward(self, X):
        a = self.adj

        G_0 = torch.mm(a, X)  # conv 1
        G_0 = self.fcl_0(G_0)  # linear layer 1
        H_0 = F.relu(G_0)
        x = torch.mm(a, H_0)  # conv 2
        return self.fcl_1(x)  # output layer

    def get_optimizer(self):
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(self.parameters(), lr=0.1, momentum=0.9)
        return criterion, optimizer

    def run_train_job(self, x_train, y_train, n_epoch):
        cross_entropy, optimizer = self.get_optimizer()

        for epoch in range(n_epoch):  # loop over the dataset multiple times

            running_loss = 0.0
            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            output = gcn.forward(x_train)
            loss = cross_entropy(output, y_train)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            print("Epoch: {}, Train Loss {:.4f}".format(epoch, loss))

        print('Finished Training')

    @staticmethod
    def normalize_adj(adj):
        return to_torch_tensor(GCN.preprocess(adj))

    @staticmethod
    def normalize_adjacency_matrix(adj):
        row_sum = np.array(adj.sum(1))  # D
        d_inv_sqrt = np.power(row_sum, -0.5).flatten()  # D -1/2
        # set infinities to 0
        d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
        d_mat = np.diag(d_inv_sqrt)  # diagonal node degree
        # i think
        # D^−1/2A^D^−1/2
        return adj.dot(d_mat).dot(d_mat)

    @staticmethod
    def preprocess(adj):
        # Ahat = A + I
        return GCN.normalize_adjacency_matrix(adj + np.eye(adj.shape[0]))


def to_torch_tensor(matrix, to_type=torch.FloatTensor):
    return torch.from_numpy(matrix).type(to_type)


if __name__ == "__main__":
    #
    A = get_adj_mtx()
    X = get_features()

    y = get_labels(n_samples=X.shape[0], class_probs=[.2, .5, .2, .1])

    X = to_torch_tensor(X)
    labels = to_torch_tensor(y, torch.LongTensor)

    gcn = GCN(adj=A, n_features=X.shape[1], n_classes=4, n_hidden=16)
    if torch.cuda.is_available():
        X = X.cuda()
        labels = labels.cuda()
        gcn = gcn.cuda()

    gcn.run_train_job(x_train=X, y_train=labels, n_epoch=250)
