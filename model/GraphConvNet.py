import torch
import torch.nn as nn
import torch.nn.functional as F
from model.get_karate_data import *
import torch.optim as optim

class GCN(nn.Module):
    """
    Graph ConvNet Model (via Kipf and Welling ICLR'2017)
    Three matrices required:
        - A: Adjacency matrix (A + I)
        - D: Diagonal matrix (D^-1/2)
        - X: Nodal feature matrix
    """
    def __init__(self, n_nodes, n_features, n_classes, h_dim_size=16):
        super(GCN, self).__init__()
        self.n_nodes = n_nodes
        self.n_features = n_features
        # fully connected layer 1
        self.fcl_0 = nn.Linear(n_features, h_dim_size, bias=True)
        # Output layer for link prediction
        self.fcl_1 = nn.Linear(h_dim_size, 2, bias=True)
        #self.out_layer = nn.Linear(8, 2, bias=True)



    def forward(self, X, A, D):
        # G = D*A*D*X*W
        G_0 = torch.mm(torch.mm(A, torch.mm(A,D)), X)
        H_0 = F.relu(self.fcl_0(G_0))

        G_1 = torch.mm(torch.mm(A, torch.mm(A,D)), H_0)
        output_logits = self.fcl_1(G_1)

        return output_logits, H_0


    def get_optimizer(self):
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(self.parameters(), lr=0.1, momentum=0.9)

        return criterion, optimizer

    def run_train_job(self, X_train, y_train, A_hat, D_hat, n_epoch):
        cross_entropy, optimizer = self.get_optimizer()

        for epoch in range(n_epoch):  # loop over the dataset multiple times

            running_loss = 0.0
            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            output, H = gcn.forward(X_train, A_hat, D_hat)
            loss = cross_entropy(output, y_train)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            print("Epoch: {}, Train Loss {:.4f}".format(epoch, loss))

        print('Finished Training')




if __name__ == "__main__":

    A = get_adj_mtx()
    X = get_features()
    W = get_weighted_graph(A.shape[0])


    n_nodes = X.shape[0]
    n_feature = X.shape[1]

    D_hat = get_degree_mtx(A)
    A_hat = get_a_hat(A)

    y = get_labels(n_samples=X.shape[0], class_probs=[.2, .5, .2, .1])

    A = torch.from_numpy(A).type(torch.FloatTensor)
    X = torch.from_numpy(X).type(torch.FloatTensor)
    D_hat = torch.from_numpy(D_hat).type(torch.FloatTensor)
    A_hat = torch.from_numpy(A_hat).type(torch.FloatTensor)
    labels = torch.from_numpy(y).type(torch.LongTensor)


    gcn = GCN(n_nodes=n_nodes, n_features=2, n_classes=4, h_dim_size=16)

    gcn.run_train_job(X_train=X, y_train=labels, A_hat=A_hat, D_hat=D_hat, n_epoch=250)