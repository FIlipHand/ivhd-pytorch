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
            eta: float = 0.1,
            device: str = "cpu",
            autoadapt=False,
            velocity_limit=False,
            finalizing_epochs = 0,
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
        self.b = 0.3
        self.device = device
        self.verbose = verbose

        self.autoadapt = autoadapt
        self.buffer_len = 10
        self.curr_max_velo = torch.tensor(([0.0]*self.buffer_len))
        self.curr_max_velo_idx = 1 
        self.velocity_limit = velocity_limit
        self.max_velocity = 1.0
        self.vel_dump = 0.95
        self.finalizing_epochs = finalizing_epochs
        self.x = None
        self.delta_x = None

    def fit_transform(self, X: torch.Tensor, NN: torch.Tensor, RN: torch.Tensor) -> torch.Tensor:
        X = X.to(self.device)
        NN = NN.to(self.device)
        RN = RN.to(self.device)
        NN = NN.reshape(-1)
        RN = RN.reshape(-1)

        if self.optimizer is None:
            return self.force_directed_method(X, NN, RN)
        else:
            return self.optimizer_method(X.shape[0], NN, RN)

    def optimizer_method(self, N,  NN, RN):
        if self.x is None:
            self.x = torch.rand((N, 1, self.n_components), requires_grad=True, device=self.device)
        optimizer = self.optimizer(params={self.x}, **self.optimizer_kwargs)
        for i in range(self.epochs):
            loss = self.__optimizer_step(optimizer, NN, RN)
            if loss < 1e-10:
                return self.x[:, 0].detach()
            if self.verbose:
                print(f"\r{i} loss: {loss.item()}, X: {self.x[0]}", end="")
                if i % 100 == 0:
                    print()

        return self.x[:, 0].detach()

    def __optimizer_step(self, optimizer, NN, RN) -> torch.tensor:
        optimizer.zero_grad()
        nn_diffs = self.x - torch.index_select(self.x, 0, NN).view(self.x.shape[0], -1, self.n_components)
        rn_diffs = self.x - torch.index_select(self.x, 0, RN).view(self.x.shape[0], -1, self.n_components)
        nn_dist = torch.sqrt(torch.sum((nn_diffs + 1e-8)*(nn_diffs + 1e-8), dim=-1, keepdim=True))
        rn_dist = torch.sqrt(torch.sum((rn_diffs + 1e-8)*(rn_diffs + 1e-8), dim=-1, keepdim=True))

        loss = torch.mean(nn_dist * nn_dist) + self.c * torch.mean((1 - rn_dist) * (1 - rn_dist))
        loss.backward()
        optimizer.step()
        return loss

    def force_directed_method(self, X: torch.Tensor, NN: torch.Tensor, RN: torch.Tensor) -> torch.Tensor:
        NN_new = NN.reshape(X.shape[0], self.nn, 1)
        NN_new = [NN_new for _ in range(self.n_components)]
        NN_new = torch.cat(NN_new, dim=-1).to(torch.long)

        RN_new = RN.reshape(X.shape[0], self.rn, 1)
        RN_new = [RN_new for _ in range(self.n_components)]
        RN_new = torch.cat(RN_new, dim=-1).to(torch.long)
        D_RN = torch.zeros(X.shape[0], self.rn)
        for i in range(self.rn):
            D_RN[i] = torch.sum((X[i] - X[RN[i]])**2, dim=-1)
        D_RN = D_RN.to(self.device)
        D_RN = D_RN.reshape(-1)
        
        if self.x is None:
            self.x = torch.rand((X.shape[0], 1, self.n_components), device=self.device)
        if self.delta_x is None:
            self.delta_x = torch.zeros_like(self.x)
        for i in range(self.epochs):
            loss = self.__force_directed_step(NN, RN, D_RN, NN_new, RN_new, finalizing=False)
            if self.verbose and i % 100 == 0:
                print(f"\r{i} loss: {loss.item()}")

        if self.finalizing_epochs > 0:
            self.max_velocity = 1000.0
            for i in range(self.finalizing_epochs):
                loss = self.__force_directed_step(NN, RN, D_RN, NN_new, RN_new, finalizing=True)
                if self.verbose and i % 100 == 0:
                    print(f"\r{i} loss: {loss.item()}")

        return self.x[:, 0]

    def __force_directed_step(self, NN, RN, D_RN, NN_new, RN_new, finalizing):
        nn_diffs = self.x - torch.index_select(self.x, 0, NN).view(self.x.shape[0], -1, self.n_components)
        rn_diffs = self.x - torch.index_select(self.x, 0, RN).view(self.x.shape[0], -1, self.n_components)
        nn_dist = torch.sqrt(torch.sum((nn_diffs+1e-8)*(nn_diffs+1e-8), dim=-1, keepdim=True))
        rn_dist = torch.sqrt(torch.sum((rn_diffs+1e-8)*(rn_diffs+1e-8), dim=-1, keepdim=True))

        f_nn, f_rn = self.__compute_forces(NN, RN, D_RN, nn_dist, rn_dist, nn_diffs, rn_diffs, NN_new, RN_new, finalizing)

        f = -f_nn - self.c*f_rn
        self.delta_x = self.a*self.delta_x + self.b*f
        
        if self.velocity_limit or self.autoadapt:
            squared_velocity = torch.sum(self.delta_x*self.delta_x, dim=-1)
            sqrt_velocity = torch.sqrt(squared_velocity)

        if self.velocity_limit:
            self.delta_x[squared_velocity > self.max_velocity**2] *= \
                self.max_velocity / sqrt_velocity[squared_velocity > self.max_velocity**2].reshape(-1, 1)

        self.x += self.eta * self.delta_x

        if self.autoadapt:   
            self.__autoadapt(sqrt_velocity)

        if self.velocity_limit:
            self.delta_x *= self.vel_dump

        loss = torch.mean(nn_dist**2) + self.c*torch.mean((1-rn_dist)**2)
        return loss

    def __autoadapt(self, sqrt_velocity):
        v_avg = self.delta_x.mean()
        self.curr_max_velo[self.curr_max_velo_idx] = sqrt_velocity.max()
        self.curr_max_velo_idx = (self.curr_max_velo_idx + 1) % self.buffer_len
        v_max = self.curr_max_velo.mean()
        if v_max > 10 * v_avg:
            self.eta /= 1.01
        elif v_max < 10 * v_avg:
            self.eta *= 1.01
        if self.eta < 0.01:
            self.eta = 0.01

    def __compute_energy(self, NN, RN, D_RN, nn_dist, rn_dist, finalizing):
        if finalizing:
            nn_energy = 0.005 / nn_dist
            rn_energy = 0.005 / rn_dist
            nn_energy[nn_dist < torch.index_select(D_RN, 0, NN).reshape(self.x.shape[0], -1, 1)] *= -1.0
            rn_energy[rn_dist < torch.index_select(D_RN, 0, RN).reshape(self.x.shape[0], -1, 1)] *= -1.0
        else:
            nn_energy = 2.0 
            rn_energy = (2.0 / rn_dist) * (rn_dist - torch.index_select(D_RN, 0, RN).reshape(self.x.shape[0], -1, 1))
        return nn_energy, rn_energy
    
    def __compute_forces(self, NN, RN, D_RN,  nn_dist, rn_dist, nn_diffs, rn_diffs, NN_new, RN_new, finalizing):
        nn_energy, rn_energy = self.__compute_energy(NN, RN, D_RN, nn_dist, rn_dist, finalizing)

        f_nn = nn_energy * nn_diffs
        f_rn = rn_energy * (rn_dist-1)/(rn_dist + 1e-8) * rn_diffs

        minus_f_nn = torch.zeros_like(f_nn).scatter_add_(src=f_nn, dim=0, index=NN_new)
        minus_f_rn = torch.zeros_like(f_rn).scatter_add_(src=f_rn, dim=0, index=RN_new)

        f_nn -= minus_f_nn
        f_rn -= minus_f_rn
        f_nn = torch.sum(f_nn, dim=1, keepdim=True)
        f_rn = torch.sum(f_rn, dim=1, keepdim=True)
        return f_nn, f_rn
