from typing import Optional, Type

import torch
from torch.optim import Optimizer


class IVHD:
    def __init__(
            self,
            n_components: int = 2,
            nn: int = 2,
            rn: int = 1,
            c: float = 0.1,
            optimizer: Optional[Type[Optimizer]] = None,
            epochs: int = 200,
            eta: float = 1.) -> None:
        self.n_components = n_components
        self.nn = nn
        self.rn = rn
        self.c = c
        self.optimizer = optimizer
        self.epochs = epochs
        self.eta = eta
        self.a = 0.2
        self.b = 1.

    def fit_transform(self, X: torch.Tensor, NN: torch.Tensor, RN: torch.Tensor) -> torch.Tensor:
        if self.optimizer is None:
            return self.force_method(X, NN, RN)
        else:
            x = torch.rand((X.shape[0], 1, self.n_components), requires_grad=True)
            x_start = x.detach().clone()
            NN = NN.reshape(-1)
            RN = RN.reshape(-1)
            optimizer = self.optimizer(params={x}, lr=10.)
            for i in range(self.epochs):
                x_copy = x.detach().clone()
                optimizer.zero_grad()
                nn_diffs = x - torch.index_select(x_copy, 0, NN).reshape(X.shape[0], -1, self.n_components)
                rn_diffs = x - torch.index_select(x_copy, 0, RN).reshape(X.shape[0], -1, self.n_components)
                nn_dist = torch.sqrt(torch.sum((nn_diffs+1e-5) ** 2, dim=-1, keepdim=True))
                rn_dist = torch.sqrt(torch.sum((rn_diffs+1e-5) ** 2, dim=-1, keepdim=True))

                loss = torch.mean(nn_dist*nn_dist) + self.c*torch.mean((1-rn_dist)*(1-rn_dist))
                print(f"\r{i} loss: {loss.item()}, X: {x[0]}", torch.mean(torch.abs(x-x_start)),
                      torch.max(torch.abs(x-x_start)), end="")
                if i % 100 == 0:
                    print()
                loss.backward()
                optimizer.step()
            return x[:, 0].detach()

    def force_method(self, X: torch.Tensor, NN: torch.Tensor, RN: torch.Tensor) -> torch.Tensor:
        x = torch.randn((X.shape[0], 1, self.n_components))
        delta_x = torch.zeros_like(x)
        NN = NN.reshape(-1)
        RN = RN.reshape(-1)
        for i in range(self.epochs):
            nn_diffs = x - torch.index_select(x, 0, NN).reshape(X.shape[0], -1, self.n_components)
            rn_diffs = x - torch.index_select(x, 0, RN).reshape(X.shape[0], -1, self.n_components)
            nn_dist = torch.sqrt(torch.sum(nn_diffs ** 2, dim=-1, keepdim=True))
            rn_dist = torch.sqrt(torch.sum(rn_diffs**2, dim=-1, keepdim=True))

            f_nn = torch.mean(nn_diffs, dim=1, keepdim=True)
            f_rn = torch.mean((1-rn_dist)/(rn_dist + 1e-5)*rn_diffs, dim=1, keepdim=True)

            loss = torch.sum(nn_dist**2) + torch.sum((1-rn_dist)**2)
            print(f"\r{i} loss: {loss.item()}", end="")

            f = -f_nn - self.c*f_rn
            delta_x = self.a*delta_x + self.b*f
            x = x + self.eta * delta_x
        return x[:, 0]
