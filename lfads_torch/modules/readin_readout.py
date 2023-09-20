import math
from glob import glob

import h5py
import numpy as np
import torch
from torch import nn


class FanInLinear(nn.Linear):
    def reset_parameters(self):
        super().reset_parameters()
        nn.init.normal_(self.weight, std=1 / math.sqrt(self.in_features))
        nn.init.constant_(self.bias, 0.0)


class MultisessionReadin(nn.ModuleList):
    def __init__(
        self,
        datafile_pattern: str,
        requires_grad: bool = False,
    ):
        modules = []
        # Identify paths that match the datafile pattern
        data_paths = sorted(glob(datafile_pattern))
        for data_path in data_paths:
            # Load the pre-computed readin transformations
            with h5py.File(data_path) as h5file:
                weight = h5file["readin_weight"][()]
                bias = -np.dot(h5file["readout_bias"][()], weight)
            # Create a linear layer and load the pre-computed parameters
            layer = nn.Linear(*weight.shape)
            layer.load_state_dict(
                {"weight": torch.tensor(weight.T), "bias": torch.tensor(bias)}
            )
            modules.append(layer)
        # Create the nn.ModuleList
        super().__init__(modules)
        # Allow the user to set requires_grad
        for param in self.parameters():
            param.requires_grad_(requires_grad)


class MultisessionReadout(nn.ModuleList):
    def __init__(
        self,
        datafile_pattern: str,
        requires_grad: bool = True,
    ):
        modules = []
        # Identify paths that match the datafile pattern
        data_paths = sorted(glob(datafile_pattern))
        for data_path in data_paths:
            # Load the pre-computed readout transformations
            with h5py.File(data_path) as h5file:
                weight = np.linalg.pinv(h5file["readin_weight"][()])
                bias = h5file["readout_bias"][()]
            # Create a linear layer and load the pre-computed parameters
            layer = nn.Linear(*weight.shape)
            layer.load_state_dict(
                {"weight": torch.tensor(weight.T), "bias": torch.tensor(bias)}
            )
            modules.append(layer)
        # Create the nn.ModuleList
        super().__init__(modules)
        # Allow the user to set requires_grad
        for param in self.parameters():
            param.requires_grad_(requires_grad)
