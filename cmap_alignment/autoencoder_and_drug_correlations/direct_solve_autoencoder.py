import torch
import torch.nn as nn
import neural_model
import numpy as np
from copy import deepcopy
from numpy.linalg import svd


# Replace the trainer.py code from
# https://github.com/uhlerlab/covid19_repurposing/tree/main/Code_autoencoding/autoencoding
# with the following:
def train_network(train_loader, test_loader):
    net = neural_model.Net()

    X = []
    for idx, batch in enumerate(train_loader):
        X.append(batch)
        print(idx, len(train_loader))
    X = torch.cat(X)
    T = X.T @ X * 1/len(X) * 1/911
    T += torch.eye(len(T))
    init = deepcopy(T)

    # Controls the degree for the autoencoder
    t = 10
    for i in range(t):
        init = init @ T

    init = init.numpy()
    # Solution from Theorem 1 for gradient flow
    U, s, Vt = svd(init)
    init = U @ np.diag(np.sqrt(s**2/2 + np.sqrt(1 + s**4/4))) @ Vt
    init = torch.from_numpy(init)

    # Custom Initialization if needed
    for idx, param in enumerate(net.parameters()):
        if idx == 0:
            param.data = init
        else:
            # Set param.data = inverse of init when decoding
            param.data.fill_(0)
    d = {}
    d['state_dict'] = net.state_dict()
    torch.save(d, 'trained_model_best.pth')
