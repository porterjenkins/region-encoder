import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions.multivariate_normal import MultivariateNormal

class AutoEncoder(nn.Module):
    """
    Denoising Autoencoder implementation
    """
    def __init__(self):
        super(AutoEncoder, self).__init__()
        ### Encoder

        # convoluional layer
        self.conv1 = nn.Conv2d(3, 6, 3)
        # Max pool layer
        self.pool = nn.MaxPool2d(2, 2)
        # Convolutional Layer
        self.conv2 = nn.Conv2d(6, 24, 3)
        # Three fully connected layers
        self.fc1 = nn.Linear(24 * 6 * 6, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 32)

        ### Decoder
        self.fc4 = nn.Linear(32, 84)
        self.fc5 = nn.Linear(84, 120)
        self.fc6 = nn.Linear(120, 24 * 6 * 6)
        self.conv3 = nn.Conv2d(24, 6, 3)
        self.up_sample3 = nn.UpsamplingBilinear2d((14, 14))
        self.conv4 = nn.Conv2d(6, 3, 3)
        self.up_sample4 = nn.UpsamplingBilinear2d((32, 32))



    def forward(self, x):
        # Encode
        #print(x.shape)
        x = self.pool(F.relu(self.conv1(x)))
        #print(x.shape)
        x = self.pool(F.relu(self.conv2(x)))
        #print(x.shape)
        x = x.view(-1, 24 * 6 * 6)
        #print(x.shape)
        x = F.relu(self.fc1(x))
        #print(x.shape)
        x = F.relu(self.fc2(x))
        #print(x.shape)

        # hidden state
        h = self.fc3(x)
        #print(self.h.shape)

        x = F.relu(self.fc4(h))
        #print(x.shape)
        x = F.relu(self.fc5(x))
        #print(x.shape)
        x = F.relu(self.fc6(x))
        #print(x.shape)

        x = x.view(-1, 24, 6, 6)
        #print(x.shape)

        x = self.up_sample3(F.relu(self.conv3(x)))
        #print(x.shape)
        x = self.up_sample4(F.relu(self.conv4(x)))
        #print(x.shape)


        return x, h


    def get_optimizer(self):
        criterion = nn.MSELoss()
        optimizer = optim.SGD(self.parameters(), lr=0.005, momentum=0.9)

        return criterion, optimizer

    def add_noise(self, image_tensor, noise_factor=.5):
        noised_image = torch.zeros(image_tensor.shape)
        batch_size = image_tensor.shape[0]
        channels = image_tensor.shape[1]
        h = image_tensor.shape[2]
        w = image_tensor.shape[3]
        mvn = MultivariateNormal(torch.zeros((h,w)), torch.eye(w))
        for i in range(batch_size):
            for k in range(channels):
                noise = torch.clamp(mvn.sample(), min=-1, max=1)
                noised_image[i, k, :, :] = image_tensor[i, k, :, :] + noise_factor*noise

        return noised_image

    def run_train_job(self, n_epoch, trainloader, print_iter=500):
        MSE, optimizer = self.get_optimizer()

        for epoch in range(n_epoch):  # loop over the dataset multiple times
            running_loss = 0.0
            for i, data in enumerate(trainloader, 0):

                inputs, labels = data
                #imshow(img=torchvision.utils.make_grid(inputs), save=True, fname='true-{}.png'.format(i))

                noisey_inputs = self.add_noise(inputs, noise_factor=.25)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward + backward + optimize
                reconstruction, h = self.forward(x=noisey_inputs)
                loss = MSE(inputs, reconstruction)
                loss.backward()
                optimizer.step()
                # print statistics
                running_loss += loss.item()
                if (i + 1) % print_iter == 0:
                    avg_loss = running_loss / print_iter
                    print("Epoch: {}, Step: {}, Train Loss {:.4f}".format(epoch, i+1, avg_loss))
                    running_loss = 0.0
                    #print(self.h)

            imshow(img=torchvision.utils.make_grid(inputs), save=True, fname='true-{}.png'.format(epoch+1))
            imshow(img=torchvision.utils.make_grid(noisey_inputs), save=True, fname='noisy-{}.png'.format(epoch+1))
            imshow(img=torchvision.utils.make_grid(reconstruction), save=True,
                   fname='reconstruction-{}.png'.format(epoch+1))




        print('Finished Training')


if __name__ == "__main__":
    ########################################################################
    # The output of torchvision datasets are PILImage images of range [0, 1].
    # We transform them to Tensors of normalized range [-1, 1].

    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    trainset = torchvision.datasets.CIFAR10(root='../tutorials/data', train=True,
                                            download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=8,
                                              shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(root='../tutorials/data', train=False,
                                           download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=8,
                                             shuffle=False, num_workers=2)

    classes = ('plane', 'car', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    ########################################################################
    # Let us show some of the training images, for fun.

    import matplotlib.pyplot as plt
    import numpy as np

    # functions to show an image


    def imshow(img, save=False, fname=None):
        img = img / 2 + 0.5     # unnormalize
        npimg = img.detach().numpy()
        plt.imshow(np.transpose(npimg, (1, 2, 0)))
        if save:
            plt.savefig("tmp/" + fname)
        else:
            plt.show()


    # get some random training images
    dataiter = iter(trainloader)
    images, labels = dataiter.next()

    auto_encoder = AutoEncoder()
    auto_encoder.run_train_job(n_epoch=25, trainloader=trainloader)
