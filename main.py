import torch
import torchvision
import matplotlib.pyplot as plt

from ivhd.ivdh import IVHD
from torch import autograd

N = 2000
NN = 5
RN = 2

if __name__ == '__main__':
    dataset = torchvision.datasets.MNIST("mnist", train=True, download=True)
    X = dataset.data[:N]
    X = X.reshape(N, -1) / 255.
    print(X.shape, torch.max(X))
    Y = dataset.targets[:N]
    ivhd = IVHD(2, 10, 2, optimizer=torch.optim.Adagrad, epochs=10000)

    rn = torch.randint(0, N, (N, RN))
    distances = torch.zeros(N, N)
    try:
        result = torch.load("mnist_dist.pt")
        print(result)
        assert result["mnist"].shape == (N, N)
        distances = result["mnist"]
    except Exception or AssertionError:
        for i in range(N):
            print(f"\r{i}", end="")
            distances[i] = torch.sum((X[i] - X)**2, dim=-1)

    torch.save({"mnist": distances}, 'mnist_dist.pt')
    #torch.load('mnist_dist.pt')

    # old way, takes too much memory
    nn = torch.zeros(N, NN, dtype=torch.int32)
    STEP = 1000
    for i in range(0, N, STEP):
        #print(i, i+STEP)
        nn[i:i+STEP, :] = torch.topk(distances[i:i+STEP, :], NN+1, dim=-1, largest=False)[1][:, 1:]
    #print(nn[:15])
    _, nn_copy = torch.topk(distances, NN + 1, dim=-1, largest=False)
    nn_copy = nn_copy[:, 1:]
    #print(nn_copy[:15])
    #print(torch.all(nn_copy == nn))
    #print(distances.shape)
    with autograd.detect_anomaly():
        x = ivhd.fit_transform(X, nn, rn)

    fig = plt.figure()
    #print(Y.shape)
    plt.title("Mnist 2d visualization")
    for i in range(10):
        points = x[Y == i]
        plt.scatter(points[:, 0], points[:, 1], label=f"{i}", marker="x", alpha=0.5)
    #print(x)
    plt.legend()
    plt.show()
