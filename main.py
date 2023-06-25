import torch
import torchvision
import matplotlib.pyplot as plt

from ivhd.ivdh import IVHD
from torch import autograd
import pandas as pd
import numpy as np
from knn_graph.faiss_generator import FaissGenerator
from knn_graph.graph import Graph
N = 60000
NN = 2
RN = 1

if __name__ == '__main__':
    dataset = torchvision.datasets.MNIST("mnist", train=True, download=True)
    X = dataset.data[:N]
    print(X.shape)
    X = X.reshape(N, -1) / 255.
    print(X.shape, torch.max(X))
    Y = dataset.targets[:N]
    ivhd = IVHD(2, NN, RN, c=0.05, eta=0.02, optimizer=None, optimizer_kwargs={"lr": 0.1}, epochs=3_000, device="cuda")

    rn = torch.randint(0, N, (N, RN))

    #faiss_generator = FaissGenerator(pd.DataFrame(X.numpy()), cosine_metric=False)
    #faiss_generator.run(nn=NN)
    #faiss_generator.save_to_binary_file("./graph_files/out.bin")
    graph = Graph()
    graph.load_from_binary_file("./graph_files/out.bin", nn_count=NN)
    nn = torch.tensor(graph.indexes.astype(np.int32))
    print(nn)

    with autograd.detect_anomaly():
        x = ivhd.fit_transform(X, nn, rn).cpu()

    fig = plt.figure()
    plt.title("Mnist 2d visualization")
    for i in range(10):
        points = x[Y == i]
        plt.scatter(points[:, 0], points[:, 1], label=f"{i}", marker=".", s=1, alpha=0.5)
    #print(x)
    plt.legend()
    plt.show()
