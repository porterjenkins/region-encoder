import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class DiscriminatorMLP(nn.Module):
    """
    Discriminator for RegionEncoder model
        - Simple Multilayer Perceptron w/ k layers
        to unify graph/image hidden states into latent
        space
    """

    def __init__(self, x_features, z_features, h_dim_size=16):
        super(DiscriminatorMLP, self).__init__()
        self.W_0 = nn.Linear(x_features + z_features, h_dim_size, bias=True)
        self.W_output = nn.Linear(h_dim_size, 2, bias=True)
        if torch.cuda.is_available():
            self.W_0 = self.W_0.cuda()
            self.W_output = self.W_output.cuda()

    def forward(self, x, z, activation=True):
        X = torch.cat((x, z), dim=-1)
        #hadamard = torch.mul(x, z)
        #h = F.relu(hadamard + self.W_0(X))
        h = F.relu(self.W_0(X))
        if activation:
            output = F.sigmoid(self.W_output(h))
        else:
            output = self.W_output(h)

        return output, h


class DiscriminatorNCF(nn.Module):
    """
    Discriminator for RegionEncoder model
        - Model based on Neural Collaborative Filtering
        paper (He et al. WWW17)
        - See eq. 11 for formulation of layer
    """

    def __init__(self, x_features, z_features):
        super(DiscriminatorNCF, self).__init__()
        pass


class DiscriminatorConcat(nn.Module):
    """
    Simple concatentation of input vectors
    """

    def forward(self, x, z):
        h = torch.cat((x, z), 0)


if __name__ == "__main__":
    n = 100
    p = 32
    mu = np.zeros(p)
    sig = np.eye(p)
    x = np.random.multivariate_normal(mean=mu, cov=sig, size=n)
    z = np.random.multivariate_normal(mean=mu, cov=sig, size=n)

    y = np.zeros(shape=(n, 1), dtype=np.float32)

    for i in range(n):
        if x[i, 0] > 0:
            y[i, 0] = np.random.binomial(1, .9)

    mod = DiscriminatorMLP(x_features=p, z_features=p, h_dim_size=32)

    x_train = torch.from_numpy(x).type(torch.FloatTensor)
    z_train = torch.from_numpy(z).type(torch.FloatTensor)
    y_train = torch.from_numpy(y).type(torch.FloatTensor)

    optimizer = optim.SGD(mod.parameters(), lr=0.1, momentum=0.9)
    cross_entropy = torch.nn.BCELoss()
    n_epoch = 100

    for i in range(n_epoch):
        optimizer.zero_grad()

        # forward + backward + optimize
        y_hat, h = mod.forward(x_train, z_train)
        loss = cross_entropy(y_hat, y_train)
        loss.backward()
        optimizer.step()

        # print statistics
        loss.item()
        print("Epoch: {}, Train Loss {:.4f}".format(i, loss.item()))
    print(h.shape)
