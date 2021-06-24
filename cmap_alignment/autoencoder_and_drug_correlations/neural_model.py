import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
import torch.nn.functional as F


class Nonlinearity(torch.nn.Module):
    def __init__(self):
        super(Nonlinearity, self).__init__()

    def forward(self, x):
        return F.leaky_relu(x)

# Replace the neural_model.py code in:
# https://github.com/uhlerlab/covid19_repurposing/tree/main/Code_autoencoding/autoencoding
# with the following.
class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        k = 911
        input_size = 911

        # Uncomment Nonlinearity() to switch to leaky relu autoencoder
        self.net = nn.Sequential(nn.Linear(input_size, k, bias=False),
                                 #Nonlinearity(),
                                 nn.Linear(k, input_size, bias=False))

    def forward(self, x):
        return self.net(x)
