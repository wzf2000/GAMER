import torch
from torch import nn
from torch.nn import functional as F


class HGNN_conv(nn.Module):
    def __init__(self, n_hid: int, bias: bool = True):
        super(HGNN_conv, self).__init__()

        self.weight = nn.Parameter(torch.Tensor(n_hid, n_hid))
        nn.init.normal_(self.weight, std=0.02)

        if bias:
            self.bias = nn.Parameter(torch.Tensor(n_hid))
            nn.init.normal_(self.bias, std=0.02)
        else:
            self.register_parameter("bias", None)

    def forward(self, x: torch.Tensor, G: torch.Tensor) -> torch.Tensor:
        x = x.matmul(self.weight)
        if self.bias is not None:
            x = x + self.bias
        x = G.matmul(x)
        return x


class HGNN(nn.Module):
    def __init__(self, n_hid: int, dropout: float = 0.2):
        super(HGNN, self).__init__()
        self.dropout = dropout
        self.hgc1 = HGNN_conv(n_hid, n_hid)
        self.hgc2 = HGNN_conv(n_hid, n_hid)

    def forward(self, x: torch.Tensor, G: torch.Tensor) -> torch.Tensor:
        x1 = self.hgc1(x, G)
        x1 = F.dropout(x1, self.dropout, training=self.training)
        x2 = self.hgc2(x1, G)
        x2 = F.dropout(x2, self.dropout, training=self.training)
        return (x1 + x2) / 2
