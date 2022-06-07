import math

# import torch
from torch import nn

# import h5py


class FanInLinear(nn.Linear):
    def reset_parameters(self):
        super().reset_parameters()
        nn.init.normal_(self.weight, std=1 / math.sqrt(self.in_features))
        nn.init.constant_(self.bias, 0.0)


# class PCRInitModuleList(nn.ModuleList):
#     def __init__(self, inits_path: str, modules: list[nn.Module]):
#         super().__init__(modules)
#         # Pull pre-computed initialization from the file, assuming correct order
#         with h5py.File(inits_path, "r") as h5file:
#             weights = [v["/" + k + "/matrix"][()] for k, v in h5file.items()]
#             biases = [v["/" + k + "/bias"][()] for k, v in h5file.items()]
#         # Load the state dict for each layer
#         for layer, weight, bias in zip(self, weights, biases):
#             state_dict = {"weight": torch.tensor(weight), "bias": torch.tensor(bias)}
#             layer.load_state_dict(state_dict)
