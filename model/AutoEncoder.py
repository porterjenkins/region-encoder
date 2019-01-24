import os
import sys
from collections import OrderedDict
import torch.nn.functional as F
import torch
import torch.nn as nn
import torch.optim as optim
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config import get_config
from model.utils import write_embeddings

class ViewEncode(nn.Module):
    def forward(self, input):
        return input.view(-1, 24 * 11 * 11)


class ViewDecode(nn.Module):
    def forward(self, input):
        return input.view(-1, 24, 11, 11)


class Tan(nn.Module):
    def forward(self, input):
        return torch.tanh(input)


class AutoEncoder(nn.Module):
    def __init__(self, img_dims=(50, 50), h_dim_size=32, cuda_override=False):
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
            ('l1', nn.Linear(24 * 11 * 11, 120)),
            ('relu3', nn.ReLU()),
            ('l2', nn.Linear(120, 84)),
            ('relu4', nn.ReLU()),
            ('l3', nn.Linear(84, h_dim_size))]
        ))

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(h_dim_size, 84),
            nn.ReLU(),
            nn.Linear(84, 120),
            nn.ReLU(),
            nn.Linear(120, 24 * 11 * 11),
            nn.ReLU(),
            ViewDecode(),
            nn.Conv2d(24, 6, 3),
            nn.ReLU(),
            nn.UpsamplingBilinear2d((14, 14)),
            nn.Conv2d(6, 3, 3),
            nn.ReLU(),
            nn.UpsamplingBilinear2d(self.img_dims),
            nn.Conv2d(3, 3, kernel_size=1),
            Tan()
        )
        if self.cuda:
            self.encoder = self.encoder.cuda()
            self.decoder = self.decoder.cuda()

    def forward(self, x, decode_only=False):

        if decode_only:
            h = x
            x = self.decoder(h)
        else:
            h = self.encoder(x)
            x = self.decoder(h)

        return x, h

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

    def run_train_job(self, n_epoch, batch_size, img_tensor, lr=.1, noise=.25):
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
                # zero the parameter gradients
                optimizer.zero_grad()
                start_idx = step * batch_size
                end_idx = start_idx + batch_size
                batch_idx = permute_idx[start_idx:end_idx]

                noisey_inputs = AutoEncoder.add_noise(img_tensor[batch_idx, :, :, :], noise_factor=noise, cuda=self.cuda)

                # forward
                reconstruction, h_batch = self.forward(x=noisey_inputs)
                loss = AutoEncoder.loss_mse(img_tensor[batch_idx, :, :, :], reconstruction)

                # Update matrix of learned representations
                hidden_state[batch_idx, :] = h_batch

                # backward
                # zero the parameter gradients
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()


                print("Epoch: {}, step: {}, Train Loss {:.4f}".format(epoch, step, loss.item()))
        print('Finished Training')

        return hidden_state


if __name__ == "__main__":

    import matplotlib.pyplot as plt
    import numpy as np
    from grid.create_grid import RegionGrid


    # functions to show an image

    def imshow(img, save=False, fname=None):
        img = img / 2 + 0.5  # unnormalize
        npimg = img.detach().numpy()
        plt.imshow(np.transpose(npimg, (1, 2, 0)))
        if save:
            plt.savefig("tmp/" + fname)
        else:
            plt.show()

    if len(sys.argv) > 1:
        epochs = int(sys.argv[1])
        learning_rate = float(sys.argv[2])
        batch_size = int(sys.argv[3])
    else:
        epochs = 25
        learning_rate = .1
        batch_size = 20

    c = get_config()
    region_grid = RegionGrid(config=c)
    region_grid.load_img_data(std_img=True)
    region_grid.img_tens_get_size()


    img_tensor = torch.Tensor(region_grid.img_tensor)
    h_dim_size = int(c['hidden_dim_size'])

    auto_encoder = AutoEncoder(img_dims=(50, 50), h_dim_size=h_dim_size)
    embedding = auto_encoder.run_train_job(n_epoch=epochs, batch_size=batch_size, img_tensor=img_tensor,
                                           lr=learning_rate)

    if torch.cuda.is_available():
        embedding = embedding.data.cpu().numpy()
    else:
        embedding = embedding.data.numpy()

    write_embeddings(arr=embedding, n_nodes=region_grid.n_regions, fname=c['autoencoder_tile_embedding_file'])
