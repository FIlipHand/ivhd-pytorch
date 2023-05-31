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
    ivhd = IVHD(2, 10, 2, optimizer=torch.optim.Adam, epochs=200)

    rn = torch.randint(0, N, (N, RN))

    faiss_generator = FaissGenerator(pd.DataFrame(X.numpy()), cosine_metric=True)
    faiss_generator.run(nn=NN)
    faiss_generator.save_to_binary_file("./graph_files/out_cosine.bin")
    graph = Graph()
    graph.load_from_binary_file("./graph_files/out.bin", nn_count=NN)
    nn = torch.tensor(graph.indexes.astype(np.int32))
    print(nn)
    # second old way
    # X_1 = X.reshape(N, 1, -1)
    # X_2 = X.reshape(1, N, -1)
    # distances = torch.zeros(N, N)
    # for i in range(N):
    #     print(f"\r{i}", end="")
    #     distances[i] = torch.sum((X[i] - X)**2, dim=-1)
    # _, nn = torch.topk(distances, NN+1, dim=-1, largest=False)
    # nn = nn[:, 1:]


    # old way, takes too much memory
    #distances = torch.sum((X_1 - X_2)**2, dim=-1)
    # _, nn = torch.topk(distances, NN+1, dim=-1, largest=False)
    # nn = nn[:, 1:]

    with autograd.detect_anomaly():
        x = ivhd.fit_transform(X, nn, rn)

    fig = plt.figure()
    print(Y.shape)
    plt.title("Mnist 2d visualization")
    for i in range(10):
        points = x[Y == i]
        plt.scatter(points[:, 0], points[:, 1], label=f"{i}", alpha=0.1)
    print(x)
    plt.legend()
    plt.show()
