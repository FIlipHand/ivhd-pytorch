from typing import Optional, Type, Dict, Any

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
            optimizer_kwargs: Dict[str, Any] = None,
            epochs: int = 200,
            eta: float = 1.,
            device: str = "cpu",
            verbose=True) -> None:
        self.n_components = n_components
        self.nn = nn
        self.rn = rn
        self.c = c
        self.optimizer = optimizer
        self.optimizer_kwargs = optimizer_kwargs
        self.epochs = epochs
        self.eta = eta
        self.a = 0.9
        self.b = 0.6
        self.device = device
        self.verbose = verbose

    def fit_transform(self, X: torch.Tensor, NN: torch.Tensor, RN: torch.Tensor) -> torch.Tensor:
        X = X.to(self.device)
        NN = NN.to(self.device)
        RN = RN.to(self.device)

        NN = NN.reshape(-1)
        RN = RN.reshape(-1)

        if self.optimizer is None:
            return self.force_method(X, NN, RN)
        else:
            return self.optimizer_method(X.shape[0], NN, RN)

    def optimizer_method(self, N,  NN, RN):
        x = torch.rand((N, 1, self.n_components), requires_grad=True, device=self.device)
        optimizer = self.optimizer(params={x}, **self.optimizer_kwargs)
        for i in range(self.epochs):
            loss = self.__optimizer_step(optimizer, x, NN, RN)
            if loss < 1e-10:
                return x[:, 0].detach()
            if self.verbose:
                print(f"\r{i} loss: {loss.item()}, X: {x[0]}", end="")
                if i % 100 == 0:
                    print()

        return x[:, 0].detach()

    def __optimizer_step(self, optimizer, x, NN, RN) -> torch.tensor:
        optimizer.zero_grad()
        nn_diffs = x - torch.index_select(x, 0, NN).view(x.shape[0], -1, self.n_components)
        rn_diffs = x - torch.index_select(x, 0, RN).view(x.shape[0], -1, self.n_components)
        nn_dist = torch.sqrt(torch.sum((nn_diffs + 1e-8)*(nn_diffs + 1e-8), dim=-1, keepdim=True))
        rn_dist = torch.sqrt(torch.sum((rn_diffs + 1e-8)*(rn_diffs + 1e-8), dim=-1, keepdim=True))

        loss = torch.mean(nn_dist * nn_dist) + self.c * torch.mean((1 - rn_dist) * (1 - rn_dist))
        loss.backward()
        optimizer.step()
        return loss

    def force_method(self, X: torch.Tensor, NN: torch.Tensor, RN: torch.Tensor) -> torch.Tensor:
        NN_new = NN.reshape(X.shape[0], self.nn, 1)
        NN_new = [NN_new for _ in range(self.n_components)]
        NN_new = torch.concatenate(NN_new, dim=-1).to(torch.long)

        RN_new = RN.reshape(X.shape[0], self.rn, 1)
        RN_new = [RN_new for _ in range(self.n_components)]
        RN_new = torch.concatenate(RN_new, dim=-1).to(torch.long)

        x = torch.rand((X.shape[0], 1, self.n_components), device=self.device)
        delta_x = torch.zeros_like(x)
        for i in range(self.epochs):
            nn_diffs = x - torch.index_select(x, 0, NN).reshape(X.shape[0], -1, self.n_components)
            rn_diffs = x - torch.index_select(x, 0, RN).reshape(X.shape[0], -1, self.n_components)
            nn_dist = torch.sqrt(torch.sum((nn_diffs+1e-8) ** 2, dim=-1, keepdim=True))
            rn_dist = torch.sqrt(torch.sum((rn_diffs+1e-8) ** 2, dim=-1, keepdim=True))
            f_nn = nn_diffs
            f_rn = (rn_dist-1)/(rn_dist + 1e-8) * rn_diffs

            minus_f_nn = torch.zeros_like(f_nn).scatter_add_(src=f_nn, dim=0, index=NN_new)
            minus_f_rn = torch.zeros_like(f_rn).scatter_add_(src=f_rn, dim=0, index=RN_new)

            f_nn -= minus_f_nn
            f_rn -= minus_f_rn
            f_nn = torch.sum(f_nn, dim=1, keepdim=True)
            f_rn = torch.sum(f_rn, dim=1, keepdim=True)

            loss = torch.mean(nn_dist**2) + self.c*torch.mean((1-rn_dist)**2)

            if self.verbose and i % 100 == 0:
                print(f"\r{i} loss: {loss.item()}")

            f = -f_nn - self.c*f_rn
            delta_x = self.a*delta_x + self.b*f
            x = x + self.eta * delta_x
        return x[:, 0]
