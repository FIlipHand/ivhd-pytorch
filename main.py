import torch
import torchvision
import numpy as np
import matplotlib.pyplot as plt

from ivhd.ivdh import IVHD

N = 1500
NN = 5
RN = 2

if __name__ == '__main__':
    dataset = torchvision.datasets.MNIST("mnist", train=True, download=True)
    X = dataset.data[:N]
    X = X.reshape(N, -1) / 255.
    print(X.shape, torch.max(X))
    Y = dataset.targets[:N]
    ivhd = IVHD(2, 10, 2)

    rn = torch.randint(0, N, (N, RN))

    X_1 = X.reshape(N, 1, -1)
    X_2 = X.reshape(1, N, -1)
    distances = torch.sum((X_1 - X_2)**2, dim=-1)
    _, nn = torch.topk(distances, NN+1, dim=-1, largest=False)
    nn = nn[:, 1:]
    print(distances.shape)
    x = ivhd.fit_transform(X, nn, rn)

    fig = plt.figure()
    print(Y.shape)
    plt.title("Mnist 2d visualization")
    for i in range(10):
        points = x[Y == i]
        plt.scatter(points[:, 0], points[:, 1], label=f"{i}")
    print(x)
    plt.legend()
    plt.show()
