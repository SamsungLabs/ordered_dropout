import numpy as np
from torch import nn
import torch.nn.functional as F

__all__ = ["ODLinear"]


class ODLinear(nn.Linear):
    def __init__(self, *args, **kwargs):
        super(ODLinear, self).__init__(*args, **kwargs)

    def forward(self, x, p=None):
        in_dim = x.size(1)  # second dimension is input dimension
        if not p:  # i.e., don't apply OD
            out_dim = self.out_features
        else:
            out_dim = int(np.ceil(self.out_features * p))
        # subsampled weights and bias
        weights_red = self.weight[:out_dim, :in_dim]
        bias_red = self.bias[:out_dim] if self.bias is not None else None
        return F.linear(x, weights_red, bias_red)
