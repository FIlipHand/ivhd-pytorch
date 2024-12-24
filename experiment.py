import argparse
import glob
import os
import re
from datetime import datetime
from pathlib import Path
from time import time
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from pacmap import PaCMAP
from sklearn.datasets import fetch_20newsgroups
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MinMaxScaler
from torch import optim
from torchvision.datasets import EMNIST, MNIST
from trimap import TRIMAP
from umap import UMAP

from ivhd.ivdh import IVHD
from knn_graph.faiss_generator import FaissGenerator
from models.tsne import MY_TSNE


def run(
    dataset=Literal["mnist", "emnist", "rcv", "amazon"],
    model=Literal["ivhd", "umap", "pacmap", "tsne", "trimap"],
    interactive=False,
    device="auto",
    graph_file="",
    save_output=False,
):
    if device == "auto":
        device = (
            torch.device("cuda")
            if torch.cuda.is_available()
            else torch.device("mps")
            if torch.backends.mps.is_available()
            else torch.device("cpu")
        )

    match dataset:
        case "mnist":
            data = MNIST("mnist", train=True, download=True)
            X = data.data
            N = X.shape[0]
            X = X.reshape(N, -1) / 255.0
            Y = data.targets[:N]
        case "emnist":
            data = EMNIST("emnist", split="balanced", train=True, download=True)
            X = data.data
            N = X.shape[0]
            X = X.reshape(N, -1) / 255.0
            Y = data.targets[:N]
        case "20ng":
            newsgroups = fetch_20newsgroups(
                data_home="20ng", subset="all", remove=("headers", "footers", "quotes")
            )
            posts = newsgroups.data
            Y = torch.Tensor(newsgroups.target)
            vectorizer = TfidfVectorizer(max_features=1000)
            tmp = vectorizer.fit_transform(posts).toarray()
            scaler = MinMaxScaler()
            tmp = scaler.fit_transform(tmp)
            pca = PCA(n_components=50)
            X = torch.Tensor(pca.fit_transform(tmp))
        case "higgs":
            if not Path("./HIGGS.csv").is_file():
                import subprocess

                subprocess.run(
                    ["wget", "https://archive.ics.uci.edu/static/public/280/higgs.zip"]
                )
                subprocess.run(["unzip", "higgs.zip"])
                subprocess.run(["gzip", "-d", "HIGGS.csv.gz"])
                pass
            X = np.loadtxt("HIGGS.csv", delimiter=",")
            Y = torch.Tensor(X[:, 0])
            X = torch.Tensor(X[:, 1:])
        case _:
            raise ValueError("only mnist, emnist, rcv1 and amazon are supported")

    match model:
        case "ivhd":
            final_file = None
            nn_param = 2
            rm_param = 1
            if not graph_file:
                # we will search for possible files
                directory = Path("./graph_files/").absolute()
                results = glob.glob(f"{Path(directory)}/{dataset}*")
                output_list = [
                    int(match.group(1))
                    for string in results
                    if (match := re.search(r"_(\d+)nn\.bin$", string))
                ]
                if output_list and max(output_list) >= nn_param:
                    final_file = results[
                        max(range(len(output_list)), key=output_list.__getitem__)
                    ]
                else:
                    final_file = str(Path(directory, f"{dataset}_{nn_param}nn.bin"))
            else:
                final_file = graph_file
            if not Path(final_file).is_file():
                faiss_generator = FaissGenerator(
                    pd.DataFrame(X.numpy()), cosine_metric=False
                )
                faiss_generator.run(nn=nn_param)
                faiss_generator.save_to_binary_file(final_file)
            model = IVHD(
                2,
                nn=nn_param,
                rn=rm_param,
                c=0.05,
                eta=0.02,
                optimizer=None,
                optimizer_kwargs={"lr": 0.1},
                epochs=3_000,
                device=device,
                velocity_limit=False,
                autoadapt=False,
                graph_file=final_file,
            )
        case "pacmap":
            model = PaCMAP(verbose=True)
        case "tsne":
            # model = TSNE(n_jobs=8, verbose=True)
            model = MY_TSNE(n_jobs=8, verbose=True)
        case "umap":
            model = UMAP(verbose=True)
        case "trimap":
            model = TRIMAP(verbose=True)
        case _:
            raise ValueError("Only support ivhd, pacmap, tsne, umap")

    # maybe we should use default timer or sth like that to be more accurate
    start = time()
    x = model.fit_transform(X)
    end = time()

    save_dir = f'./experiments/{dataset}_{datetime.now().strftime("%Y-%m-%d_%H:%M:%S")}'
    os.makedirs(exist_ok=True, name=save_dir)

    if save_output:
        data = np.hstack((x, Y.numpy().reshape(-1, 1)))
        np.savetxt(
            str(Path(save_dir, "reduced.txt")),
            data,
            delimiter=",",
            header="Feature1,Feature2,Label",
            comments="",
            fmt="%.6f",
        )

    with open(str(Path(save_dir, "timing.txt")), "w") as f:
        f.write(str(end - start))

    fig = plt.figure(figsize=(16, 8))
    plt.title(f"{dataset} 2d visualization")

    sns.scatterplot(
        x=x[:, 0],
        y=x[:, 1],
        hue=Y,
        s=2,
        palette=sns.color_palette("tab10", torch.unique(Y).size(0)),
    )
    plt.legend()
    if interactive:
        plt.show()
    else:
        plt.savefig(str(Path(save_dir, "image.png")))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run the dataset and model configuration."
    )

    parser.add_argument(
        "--dataset",
        type=str,
        choices=["mnist", "emnist", "20ng", "higgs"],
        required=True,
        help="Specify the dataset to use. Choices are: 'mnist', 'emnist', '20ng', 'higgs'.",
    )

    parser.add_argument(
        "--model",
        type=str,
        choices=["ivhd", "umap", "pacmap", "tsne", "trimap"],
        required=True,
        help="Specify the model to use. Choices are: 'ivhd', 'umap', 'pacmap', 'tsne', 'trimap'.",
    )

    parser.add_argument("--graph", type=str, dest="graph")

    parser.add_argument(
        "--interactive",
        action="store_true",
        dest="interactive",
        help="Anable interactive mode. Save the plot instead of displaying it.",
    )

    parser.add_argument("--save-output", action="store_true", dest="save_output")

    args = parser.parse_args()

    run(
        dataset=args.dataset,
        model=args.model,
        interactive=args.interactive,
        graph_file=args.graph,
        save_output=args.save_output,
    )
