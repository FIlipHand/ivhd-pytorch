import torch
import torchvision
import matplotlib.pyplot as plt

from ivhd.ivdh import IVHD
from torch import autograd
import pandas as pd
import numpy as np
from knn_graph.faiss_generator import FaissGenerator
from knn_graph.graph import Graph
import os
N = 60000
NN = 2
RN = 1
DATASET_NAME = "mnist"


if __name__ == '__main__':
    dataset = torchvision.datasets.MNIST("mnist", train=True, download=True)
    X = dataset.data[:N]
    X = X.reshape(N, -1) / 255.
    Y = dataset.targets[:N]
    ivhd = IVHD(2, NN, RN, c=0.05, eta=0.02, optimizer=None, optimizer_kwargs={"lr": 0.1},
                epochs=3_000, device="cuda", velocity_limit=False, autoadapt=False)

    rn = torch.randint(0, N, (N, RN))

    nn_path = f"./graph_files/{DATASET_NAME}_{NN}nn.bin"

    if not os.path.exists(nn_path):
        faiss_generator = FaissGenerator(pd.DataFrame(X.numpy()), cosine_metric=False)
        faiss_generator.run(nn=NN)
        faiss_generator.save_to_binary_file(nn_path)

    graph = Graph()
    graph.load_from_binary_file("./graph_files/out.bin", nn_count=NN)
    nn = torch.tensor(graph.indexes.astype(np.int32))
    d = torch.zeros(N, RN)
    for i in range(RN):
        d[i] = torch.sum((X[i] - X[rn[i]])**2, dim=-1)

    fig = plt.figure(figsize=(16, 8))
    plt.title("Mnist 2d visualization")
    axes = fig.subplots(nrows=1, ncols=2)

    x = ivhd.fit_transform(X, nn, rn, d).cpu()

    for i in range(10):
        points = x[Y == i]
        axes[0].scatter(points[:, 0], points[:, 1], label=f"{i}", marker=".", s=1, alpha=0.5)
    #print(x)
    axes[0].legend()
    ivhd.epochs = 50
    x = ivhd.fit_transform(X, nn, rn, d, finalizing=True).cpu()

    for i in range(10):
        points = x[Y == i]
        axes[1].scatter(points[:, 0], points[:, 1], label=f"{i}", marker=".", s=1, alpha=0.5)
    #print(x)
    axes[1].legend()
    plt.show()
