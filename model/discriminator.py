import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F

class DiscriminatorMLP(nn.Module):
    """
    Discriminator for RegionEncoder model
        - Simple Multilayer Perceptron w/ k layers
        to unify graph/image hidden states into latent
        space
    """
    def __init__(self, x_features, z_features):
        super(DiscriminatorMLP, self).__init__()
        self.x_features = x_features
        self.z_features = z_features
        self.W_0 = nn.Linear(x_features + z_features, 16, bias=True)
        self.W_output = nn.Linear(16, 1, bias=True)

    def forward(self, X):
        h = F.relu(self.W_0(X))
        y_hat = F.sigmoid(self.W_output(h))

        return y_hat

class DisciminatorNCF(nn.Module):
    """
    Discriminator for RegionEncoder model
        - Model based on Neural Collaborative Filtering
        paper (He et al. WWW17)
        - See eq. 11 for formulation of layer
    """
    def __init__(self):
        super(DisciminatorNCF, self).__init__()
        pass

