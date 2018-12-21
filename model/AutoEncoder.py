import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class AutoEncoder(nn.Module):
    """
    Denoising Autoencoder implementation
    """
    def __init__(self):
        super(AutoEncoder, self).__init__()
        ### Encoder

        # convoluional layer
        self.conv1 = nn.Conv2d(3, 6, 5)
        # Max pool layer
        self.pool = nn.MaxPool2d(2, 2)
        # Convolutional Layer
        self.conv2 = nn.Conv2d(6, 16, 5)
        # Three fully connected layers
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 32)

        ### Decoder
        self.fc4 = nn.Linear(32, 84)
        self.fc5 = nn.Linear(84, 120)
        self.fc6 = nn.Linear(120, 16 * 5 * 5)
        self.conv3 = nn.Conv2d(6, 16, 5)
        self.up_sample3 = nn.UpsamplingBilinear2d(14, 14)
        self.conv4 = nn.Conv2d(3, 6, 5)
        self.up_sample4 = nn.UpsamplingBilinear2d(32, 32)







    def forward(self, x):
        # Encode
        print(x.shape)
        x = self.pool(F.relu(self.conv1(x)))
        print(x.shape)
        x = self.pool(F.relu(self.conv2(x)))
        print(x.shape)
        x = x.view(-1, 16 * 5 * 5)
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

        x = x.view(4, 16, 5, 5)
        print(x.shape)

        x = self.up_sample3(F.relu(self.conv3(x)))
        print(x.shape)
        x = self.up_sample4(F.relu(self.conv4(x)))
        print(x.shape)


        return x




########################################################################
# The output of torchvision datasets are PILImage images of range [0, 1].
# We transform them to Tensors of normalized range [-1, 1].

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

trainset = torchvision.datasets.CIFAR10(root='./tutorials/data', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
                                          shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./tutorials/data', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=4,
                                         shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

########################################################################
# Let us show some of the training images, for fun.

import matplotlib.pyplot as plt
import numpy as np

# functions to show an image


def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


# get some random training images
dataiter = iter(trainloader)
images, labels = dataiter.next()

ae = AutoEncoder()
ae.forward(images)

# show images
#print(' '.join('%5s' % classes[labels[j]] for j in range(4)))
#imshow(torchvision.utils.make_grid(images))




#print(images.shape)