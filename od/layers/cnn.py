import numpy as np
from torch import nn

__all__ = ["ODConv1d", "ODConv2d", "ODConv3d"]


def od_conv_forward(layer, x, p=None):
    in_dim = x.size(1)  # second dimension is input dimension
    if not p:  # i.e., don't apply OD
        out_dim = layer.out_channels
    else:
        out_dim = int(np.ceil(layer.out_channels * p))
    # subsampled weights and bias
    weights_red = layer.weight[:out_dim, :in_dim]
    bias_red = layer.bias[:out_dim] if layer.bias is not None else None
    return layer._conv_forward(x, weights_red, bias_red)


class ODConv1d(nn.Conv1d):
    def __init__(self, *args, **kwargs):
        super(ODConv1d, self).__init__(*args, **kwargs)

    def forward(self, x, p=None):
        return od_conv_forward(self, x, p)


class ODConv2d(nn.Conv2d):
    def __init__(self, *args, **kwargs):
        super(ODConv2d, self).__init__(*args, **kwargs)

    def forward(self, x, p=None):
        return od_conv_forward(self, x, p)


class ODConv3d(nn.Conv3d):
    def __init__(self, *args, **kwargs):
        super(ODConv3d, self).__init__(*args, **kwargs)

    def forward(self, x, p=None):
        return od_conv_forward(self, x, p)
