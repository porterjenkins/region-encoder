import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch.optim as optim

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



class RegionEncoder(nn.Module):
    """
    Implementatino of proposed model for
    Multi-Modal Region Encoding (MMRE)
    """
    def __init__(self):
        super(RegionEncoder, self).__init__()
        self.l_1 = LinearLayer_1(8, 4)
        self.l_2 = LinearLayer_2(4, 1)

    def forward(self, X):
        h = self.l_1.forward(X)
        y_hat = self.l_2.forward(h)

        return y_hat


n = 100
p = 8
mu = np.zeros(p)
sig = np.eye(p)
X = np.random.multivariate_normal(mean=mu, cov=sig, size=n)

y = np.zeros(shape=(n,1), dtype=np.float32)

for i in range(n):
    if X[i, 0] > 0:
        y[i,0] = np.random.binomial(1, .9)


mod = RegionEncoder()

X_train = torch.from_numpy(X).type(torch.FloatTensor)
y_train = torch.from_numpy(y).type(torch.FloatTensor)


optimizer = optim.SGD(mod.parameters(), lr=0.1, momentum=0.9)
cross_entropy = torch.nn.BCELoss()
n_epoch = 100

for i in range(n_epoch):

    optimizer.zero_grad()

    # forward + backward + optimize
    y_hat = mod.forward(X_train)
    loss = cross_entropy(y_hat, y_train)
    loss.backward()
    optimizer.step()

    # print statistics
    loss.item()
    print("Epoch: {}, Train Loss {:.4f}".format(i, loss.item()))