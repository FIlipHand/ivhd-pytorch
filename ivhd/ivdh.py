import torch


class IVHD:
    def __init__(
            self,
            n_components: int = 2,
            nn: int = 2,
            rn: int = 1,
            c: float = 0.1,
            epochs: int = 200,
            eta: float = 1.) -> None:
        self.n_components = n_components
        self.nn = nn
        self.rn = rn
        self.c = c
        self.epochs = epochs
        self.eta = eta
        self.a = 0.2
        self.b = 1.

    def fit_transform(self, X: torch.Tensor, NN: torch.Tensor, RN: torch.Tensor):
        x = torch.randn((X.shape[0], 1, self.n_components))
        delta_x = torch.zeros_like(x)
        NN = NN.reshape(-1)
        RN = RN.reshape(-1)
        for i in range(self.epochs):
            print(f"\r{i}", end="")
            nn_diffs = x - torch.index_select(x, 0, NN).reshape(X.shape[0], -1, self.n_components)
            rn_diffs = x - torch.index_select(x, 0, RN).reshape(X.shape[0], -1, self.n_components)
            rn_dist = torch.sqrt(torch.sum(rn_diffs**2, dim=-1, keepdim=True))

            f_nn = torch.mean(nn_diffs, dim=1, keepdim=True)
            f_rn = torch.mean((1-rn_dist)/(rn_dist + 1e-5)*rn_diffs, dim=1, keepdim=True)

            f = -f_nn + self.c*f_rn
            delta_x = self.a*delta_x + self.b*f
            x = x + self.eta * delta_x
        return x[:, 0]
