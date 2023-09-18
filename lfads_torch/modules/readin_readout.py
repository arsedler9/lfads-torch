import math

import h5py
from torch import nn


class FanInLinear(nn.Linear):
    def reset_parameters(self):
        super().reset_parameters()
        nn.init.normal_(self.weight, std=1 / math.sqrt(self.in_features))
        nn.init.constant_(self.bias, 0.0)


class HDF5InitLinear(nn.Linear):
    def __init__(self, inits_path: str, *args, **kwargs):
        super().__init__(*args, **kwargs)
        with h5py.File(inits_path) as h5file:
            state_dict = {
                "weight": h5file["readin_init_weight"][()],
                "bias": h5file["readin_init_bias"][()],
            }
        self.load_state_dict(state_dict)
