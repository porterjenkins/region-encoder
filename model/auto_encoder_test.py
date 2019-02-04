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



class AutoEncoder(nn.Module):
    """
    CPU-only implementation of AutoEncoder used for testing architectures
    """
    def __init__(self, img_dims, h_dim_size=32):
        super(AutoEncoder, self).__init__()
        self.img_dims = img_dims
        self.h_dim_size = h_dim_size
        ### Encoder

        # convoluional layer
        self.conv1 = nn.Conv2d(3, 6, 3)
        # Max pool layer
        self.pool = nn.MaxPool2d(2, 2)
        # Convolutional Layer
        self.conv2 = nn.Conv2d(6, 24, 3)
        # Three fully connected layers
        self.fc1 = nn.Linear(24 * 11 * 11, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, h_dim_size)

        ### Decoder
        self.fc4 = nn.Linear(h_dim_size, 84)
        self.fc5 = nn.Linear(84, 120)
        self.fc6 = nn.Linear(120, 24 * 11 * 11)
        self.conv3 = nn.Conv2d(24, 6, 3)
        self.up_sample3 = nn.UpsamplingBilinear2d((14, 14))
        self.conv4 = nn.Conv2d(6, 3, 3)
        self.up_sample4 = nn.UpsamplingBilinear2d(self.img_dims)
        self.conv5 = nn.Conv2d(3, 3, kernel_size=1)



    def forward(self, x):
        # Encode
        print(x.shape)
        x = self.pool(F.relu(self.conv1(x)))
        print(x.shape)
        x = self.pool(F.relu(self.conv2(x)))
        print(x.shape)
        x = x.view(-1, 24 * 11 * 11)
        print(x.shape)
        x = F.relu(self.fc1(x))
        print(x.shape)
        x = F.relu(self.fc2(x))
        print(x.shape)

        # hidden state
        h = self.fc3(x)
        print(h.shape)

        x = F.relu(self.fc4(h))
        print(x.shape)
        x = F.relu(self.fc5(x))
        print(x.shape)
        x = F.relu(self.fc6(x))
        print(x.shape)

        x = x.view(-1, 24, 11, 11)
        print(x.shape)

        x = self.up_sample3(F.relu(self.conv3(x)))
        print(x.shape)
        x = self.up_sample4(F.relu(self.conv4(x)))
        print(x.shape)
        x = torch.tanh(self.conv5(x))
        print(x.shape)

        return x, h


    def get_optimizer(self, lr):
        optimizer = optim.SGD(self.parameters(), lr=lr, momentum=0.9)

        return optimizer

    @staticmethod
    def add_noise(image_tensor, noise_factor=.5):

        batch_size = image_tensor.shape[0]
        channels = image_tensor.shape[1]
        h = image_tensor.shape[2]
        w = image_tensor.shape[3]
        noise = torch.randn((batch_size, channels, h, w))
        noised_image = image_tensor + noise_factor * noise

        return noised_image

    @staticmethod
    def loss_mse(img_input, img_reconstruction):
        err = img_input - img_reconstruction
        mse = torch.mean(torch.pow(err, 2))

        return mse

    def run_train_job(self, n_epoch, img_tensor, lr=.1, noise=.25):
        optimizer = self.get_optimizer(lr)
        n_samples = img_tensor.shape[0]

        hidden_state = torch.zeros(n_samples, self.h_dim_size)

        batch_size = 5
        for epoch in range(n_epoch):  # loop over the dataset multiple times
            permute_idx = np.random.permutation(np.arange(n_samples))
            for step in range(int(n_samples / batch_size)):
                start_idx = step * batch_size
                end_idx = start_idx + batch_size
                batch_idx = permute_idx[start_idx:end_idx]

                noisey_inputs = AutoEncoder.add_noise(img_tensor[batch_idx, :, :, :], noise_factor=noise)

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
    else:
        epochs = 25
        learning_rate = .1

    c = get_config()
    region_grid = RegionGrid(config=c)
    region_grid.load_img_data(std_img=True)


    img_tensor = torch.Tensor(region_grid.img_tensor)
    h_dim_size = int(c['hidden_dim_size'])

    auto_encoder = AutoEncoder(img_dims=(50, 50), h_dim_size=h_dim_size)
    embedding = auto_encoder.run_train_job(n_epoch=epochs, img_tensor=img_tensor, lr=learning_rate)