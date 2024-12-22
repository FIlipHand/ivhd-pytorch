from typing import Literal

from ivhd.ivdh import IVHD
from models.tsne import MY_TSNE
from umap import UMAP
from pacmap import PaCMAP
from torchvision.datasets import MNIST, EMNIST
from sklearn.datasets import fetch_rcv1
import matplotlib.pyplot as plt
import argparse
import glob
from pathlib import Path
import re
from datetime import datetime
import torch

device = (
    torch.device("cuda") if torch.cuda.is_available() else
    torch.device("mps") if torch.backends.mps.is_available() else
    torch.device("cpu")
)

def run(dataset=Literal['mnist', 'emnist', 'rcv', 'amazon'],
        model=Literal["ivhd", "umap", "pacmap", "tsne"],
        interactive=False):

    match dataset:
        case 'mnist':
            data = MNIST("mnist", train=True, download=True)
            X = data.data
            N = X.shape[0]
            X = X.reshape(N, -1) / 255.
            Y = data.targets[:N]
        case 'emnist':
            data = EMNIST("emnist", split="balanced", train=True, download=True)
            X = data.data
            N = X.shape[0]
            X = X.reshape(N, -1) / 255.
            Y = data.targets[:N]
        case 'rcv':
            data = fetch_rcv1(data_home='rcv')
            X = data.data
            N = X.shape[0]
            X = X.reshape(N, -1) / 255.
            Y = data.target[:N]
            print(X.iloc[:3, :20])
            exit()
        case 'amazon':
            raise NotImplementedError('Not yet implemented!')
        case _:
            raise ValueError('only mnist, emnist, rcv1 and amazon are supported')


    match model:
        case 'ivhd':
            def choose_graph_file():
                directory = str(Path('./graph_files/').absolute())
                results = glob.glob(f'{directory}/{dataset}*')
                output_list = [
                    int(match.group(1)) for string in results
                    if (match := re.search(r'_(\d+)nn\.bin$', string))
                ]
                if output_list:
                    return results[max(range(len(output_list)), key=output_list.__getitem__)]
                else:
                    return None

            model = IVHD(2, c=0.05, eta=0.02, optimizer=None, optimizer_kwargs={"lr": 0.1},
                        epochs=3_000, device=device, velocity_limit=False, autoadapt=False,
                        graph_file=choose_graph_file())
        case 'pacmap':
            model = PaCMAP(verbose=True)
        case 'tsne':
            # model = TSNE(n_jobs=8, verbose=True)
            model = MY_TSNE(n_jobs=8, verbose=True)
        case 'umap':
            model = UMAP(verbose=True)
        case _:
            raise ValueError("Only support ivhd, pacmap, tsne, umap")

    # if isinstance(model, TSNE): # eh eh eh
    #     x = model.fit(X)
    # else:
    x = model.fit_transform(X)

    fig = plt.figure(figsize=(16, 8))
    plt.title(f"{dataset} 2d visualization")

    for i in range(10):
        points = x[Y == i]
        plt.scatter(points[:, 0], points[:, 1], label=f"{i}", marker=".", s=1, alpha=0.5)
    plt.legend()
    if not interactive: #?
        plt.show()
    else:
        plt.savefig(f'./{dataset}_{datetime.now().strftime("%Y-%m-%d_%H:%M:%S")}')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the dataset and model configuration.")

    parser.add_argument(
        '--dataset',
        type=str,
        choices=['mnist', 'emnist', 'rcv', 'amazon'],
        required=True,
        help="Specify the dataset to use. Choices are: 'mnist', 'emnist', 'rcv1', 'amazon'."
    )

    parser.add_argument(
        '--model',
        type=str,
        choices=['ivhd', 'umap', 'pacmap', 'tsne'],
        required=True,
        help="Specify the model to use. Choices are: 'ivhd', 'umap', 'pacmap', 'tsne'."
    )

    parser.add_argument(
        '--interactive',
        action='store_false',
        dest='interactive',
        help="Anable interactive mode. Save the plot instead of displaying it."
    )

    args = parser.parse_args()

    run(dataset=args.dataset, model=args.model, interactive=args.interactive)
