import math

import h5py
import numpy as np
from torch import nn


class FanInLinear(nn.Linear):
    def reset_parameters(self):
        super().reset_parameters()
        nn.init.normal_(self.weight, std=1 / math.sqrt(self.in_features))
        nn.init.constant_(self.bias, 0.0)


class MultisessionReadin(nn.Linear):
    def __init__(
        self,
        params_path: str,
        requires_grad: bool = False,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        with h5py.File(params_path) as h5file:
            weight = h5file["readin_weight"][()]
            bias = -np.dot(h5file["readout_bias"][()], weight)
        self.load_state_dict({"weight": weight, "bias": bias})
        self.requires_grad = requires_grad


class MultisessionReadout(nn.Linear):
    def __init__(
        self,
        params_path: str,
        requires_grad: bool = True,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        with h5py.File(params_path) as h5file:
            weight = np.linalg.pinv(h5file["readin_weight"][()])
            bias = h5file["readout_bias"][()]
        self.load_state_dict({"weight": weight, "bias": bias})
        self.requires_grad = requires_grad
