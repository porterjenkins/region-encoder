import os
import sys
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config import get_config
from model.utils import write_embeddings


class ViewEncode(nn.Module):
    def forward(self, input):
        return input.view(-1, 24 * 48 * 48)


class Tan(nn.Module):
    def forward(self, input):
        return torch.tanh(input)


class AutoEncoder(nn.Module):
    def __init__(self, img_dims=(200, 200), h_dim_size=32, cuda_override=False):
        super(AutoEncoder, self).__init__()
        self.cuda = torch.cuda.is_available() and not cuda_override
        print(f"Cuda Set to {self.cuda}")
        self.h_dim_size = h_dim_size
        self.img_dims = img_dims
        # Encoder
        self.encoder = nn.Sequential(OrderedDict([
            ('conv', nn.Conv2d(3, 6, 3)),
            ('relu1', nn.ReLU()),
            ('pool', nn.MaxPool2d(2, 2)),
            ('conv2', nn.Conv2d(6, 24, 3)),
            ('relu2', nn.ReLU()),
            ('pool2', nn.MaxPool2d(2, 2)),
            ('view', ViewEncode()),
            ('l1', nn.Linear(24 * 48 * 48, 120)),
            ('relu3', nn.ReLU()),
            ('l2', nn.Linear(120, 84)),
            ('relu4', nn.ReLU()),
            ('l3', nn.Linear(84, h_dim_size))]
        ))

        if self.cuda:
            self.encoder = self.encoder.cuda()

    def forward(self, x, neighbor, distance):

        h = self.encoder(x)
        h_n = self.encoder(neighbor)
        h_d = self.encoder(distance)
        return h, h_n, h_d

    def get_optimizer(self, lr):
        optimizer = optim.SGD(self.parameters(), lr=lr, momentum=0.9)

        return optimizer

    @staticmethod
    def add_noise(image_tensor, noise_factor=.5, cuda=False):

        batch_size = image_tensor.shape[0]
        channels = image_tensor.shape[1]
        h = image_tensor.shape[2]
        w = image_tensor.shape[3]
        noise = torch.randn((batch_size, channels, h, w))
        if cuda:
            noise = noise.cuda()
        noised_image = image_tensor + noise_factor * noise

        return noised_image

    @staticmethod
    def loss_mse(img_input, img_reconstruction):
        err = img_input - img_reconstruction
        mse = torch.mean(torch.pow(err, 2))

        return mse

    @staticmethod
    def triplet_loss(patch, neighbor, distant, l=1, m=0.1):
        l_n = torch.norm(patch - neighbor, dim=1)
        l_d = torch.norm(patch - distant, dim=1)
        l_nd = l_n - l_d
        loss_ind = F.relu(l_nd + m)
        penalty = (torch.norm(patch, dim=1) + torch.norm(neighbor, dim=1) + torch.norm(distant, dim=1))
        loss_ind_penalty = loss_ind + l * penalty
        loss = torch.sum(loss_ind_penalty)
        return loss

    def run_train_job(self, n_epoch, img_tensor, lr=.1, batch_size=25):
        if self.cuda:
            img_tensor = img_tensor.cuda()
        optimizer = self.get_optimizer(lr)
        n_samples = img_tensor.shape[0]

        hidden_state = torch.zeros(n_samples, self.h_dim_size)
        if self.cuda:
            hidden_state = hidden_state.cuda()


        for epoch in range(n_epoch):  # loop over the dataset multiple times
            permute_idx = np.random.permutation(np.arange(n_samples))
            for step in range(int(n_samples / batch_size)):
                start_idx = step * batch_size
                end_idx = start_idx + batch_size
                batch_idx = permute_idx[start_idx:end_idx]

                # patch, neighbor, distant
                triplets = region_grid.create_triplets(batch_idx, img_tensor)

                # forward
                h_batch, neighbor, distance = self.forward(x=triplets[0], neighbor=triplets[1], distance=triplets[0])

                loss = self.triplet_loss(h_batch, neighbor, distance)

                # Update matrix of learned representations
                hidden_state[batch_idx, :] = h_batch[0]

                # backward
                # zero the parameter gradients
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                print("Epoch: {}, step: {}, Train Loss {:.4f}".format(epoch, step, loss.item()))
        print('Finished Training')

        return hidden_state


if __name__ == "__main__":

    import numpy as np
    from grid.create_grid import RegionGrid

    if len(sys.argv) > 1:
        epochs = int(sys.argv[1])
        learning_rate = float(sys.argv[2])
    else:
        epochs = 25
        learning_rate = .05

    c = get_config()
    region_grid = RegionGrid(config=c)
    region_grid.load_img_data(std_img=True)
    region_grid.img_tens_get_size()

    img_tensor = torch.Tensor(region_grid.img_tensor)
    h_dim_size = int(c['hidden_dim_size'])

    auto_encoder = AutoEncoder(img_dims=(200, 200), h_dim_size=h_dim_size)
    embedding = auto_encoder.run_train_job(n_epoch=epochs, img_tensor=img_tensor, lr=learning_rate)

    if torch.cuda.is_available():
        embedding = embedding.data.cpu().numpy()
    else:
        embedding = embedding.data.numpy()

    write_embeddings(arr=embedding, n_nodes=region_grid.n_regions, fname=c['autoencoder_embedding_file'])
