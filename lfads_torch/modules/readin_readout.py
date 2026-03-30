import abc
import math
from glob import glob

import h5py
import numpy as np
import torch
from torch import nn
import geotorch
from typing import Optional


class FanInLinear(nn.Linear):
    def reset_parameters(self):
        super().reset_parameters()
        nn.init.normal_(self.weight, std=1 / math.sqrt(self.in_features))
        nn.init.constant_(self.bias, 0.0)

class MLPReadout(nn.Module):
    def __init__(self, in_features, hidden_size, num_layers, out_features):
        super(MLPReadout, self).__init__()
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            if i == 0:
                self.layers.append(nn.Linear(in_features, hidden_size))
            else:
                self.layers.append(nn.Linear(hidden_size, hidden_size))
        self.output_layer = nn.Linear(hidden_size, out_features)
    def forward(self, x):
        for layer in self.layers:
            x = torch.relu(layer(x))
        x = self.output_layer(x)
        return x

def orthogonal_matrix(n):
    # Random orthogonal via QR
    A = torch.randn(n, n)
    Q, _ = torch.linalg.qr(A)
    return Q

class AdditiveCoupling(nn.Module):
    """
    NICE-style additive coupling: split along features into (a, b),
    update b <- b + m(a). Inverse: b <- b - m(a). Volume-preserving.
    """
    def __init__(self, dim, hidden, num_layers):
        super().__init__()
        self.dim = dim
        self.split = dim // 2
        # Simple MLP for m(a)
        layers = []
        in_f = self.split
        for _ in range(num_layers - 1):
            layers += [nn.Linear(in_f, hidden), nn.ReLU()]
            in_f = hidden
        layers += [nn.Linear(in_f, dim - self.split)]
        self.m = nn.Sequential(*layers)

    def forward(self, x, reverse=False):
        a, b = x[..., :self.split], x[..., self.split:]
        shift = self.m(a)
        if reverse:
            b = b - shift
        else:
            b = b + shift
        return torch.cat([a, b], dim=-1)

class OrthoPerm(nn.Module):
    """
    Orthogonal permutation (learnable). Determinant ±1, volume-preserving.
    """
    def __init__(self, dim):
        super().__init__()
        Q = orthogonal_matrix(dim)
        self.Q = nn.Parameter(Q)  # can freeze for strict const det
    def forward(self, x, reverse=False):
        if reverse:
            return x @ self.Q.transpose(-1, -2)
        else:
            return x @ self.Q
        
class InjectiveFlowReadout(nn.Module):
    """
    Injective, volume-preserving readout:
      x = W z + b  (Stiefel embed to N dims)
      then K steps of [OrthoPerm -> AdditiveCoupling]  (each vol-preserving)
    forward:  Z[B,T,D] -> Y[B,T,N]
    reverse:  Y[B,N]   -> z[B,D]  (per-time inverse)
    """
    def __init__(self, in_features, out_features, hidden_size=256, num_layers=3, num_steps=6):
        super().__init__()
        assert out_features >= in_features, "out_features must be >= in_features"
        self.D = in_features
        self.N = out_features
        self.num_steps = num_steps

        # 1) Stiefel embedding to lift D→N (injective linear; exact left-inverse via W^T)
        self.embed = StiefelLinear(in_features, out_features, use_bias=True)

        # 2) Volume-preserving flow over R^N
        steps = []
        for _ in range(num_steps):
            steps += [OrthoPerm(self.N), AdditiveCoupling(self.N, hidden_size, num_layers)]
        self.flow = nn.ModuleList(steps)

    def forward(self, Z, reverse=False):
        """
        If reverse=False: Z[B,T,D] -> Y[B,T,N]
        If reverse=True:  Z[B,N]   -> z[B,D]  (inverse)
        """
        if not reverse:
            assert Z.dim() == 3 and Z.size(-1) == self.D, "Expected [B,T,D]"
            B, T, _ = Z.shape
            X = self.embed(Z.reshape(-1, self.D))   # [B*T, N]
            for mod in self.flow:
                X = mod(X, reverse=False)
            Y = X.view(B, T, self.N)
            return Y
        else:
            # inverse used per-time or aggregated: accepts [B,N]
            assert Z.dim() == 2 and Z.size(-1) == self.N, "Expected [B,N] for inverse"
            X = Z
            for mod in reversed(self.flow):
                X = mod(X, reverse=True)
            z = self.embed.left_inverse(X)          # [B, D]
            return z
        
class _MultisessionModuleList(abc.ABC, nn.ModuleList):
    def __init__(
        self,
        datafile_pattern: str,
        pcr_init: bool,
        requires_grad: bool,
        module: str, # 'linear' or 'injective'
    ):
        modules = []
        # Identify paths that match the datafile pattern
        data_paths = sorted(glob(datafile_pattern))
        for data_path in data_paths:
            if pcr_init:
                # Load the pre-computed transformations
                state_dict = self._get_state_dict(data_path)
                out_features, in_features = state_dict["weight"].shape
                if module == 'linear':
                    layer = nn.Linear(in_features, out_features)
                    layer.load_state_dict(state_dict)
                else:
                    ValueError('pcr_init only works with linear layer')
            else:
                if module == 'linear':
                    # Infer only the input dimension from the file
                    in_features, out_features = self._get_layer_shape(data_path)
                    layer = nn.Linear(in_features, out_features)
                elif module == 'injective':
                    in_features, out_features = self._get_layer_shape(data_path)
                    layer = MultiCompartmentReadoutParametrized(
                        data_path=data_path,
                        in_features=in_features,
                    )
                elif module == 'stiefel':
                    in_features, out_features = self._get_layer_shape(data_path)
                    layer = StiefelLinear(in_features, out_features)
            modules.append(layer)
        # Create the nn.ModuleList
        super().__init__(modules)
        # Allow the user to set requires_grad
        for param in self.parameters():
            param.requires_grad_(requires_grad)

    @abc.abstractmethod
    def _get_layer_shape(self, data_path):
        pass

    @abc.abstractmethod
    def _get_state_dict(self, data_path):
        pass


class MultisessionReadin(_MultisessionModuleList):
    def __init__(
        self,
        datafile_pattern: str,
        out_features: int = None,
        pcr_init: bool = True,
        requires_grad: bool = False,
        module: str = 'linear',
    ):
        assert (
            out_features is not None
        ) != pcr_init, "Setting `out_features` mutually excludes `pcr_init`."
        self.out_features = out_features
        super().__init__(
            datafile_pattern=datafile_pattern,
            pcr_init=pcr_init,
            requires_grad=requires_grad,
            module=module,
        )

    def _get_layer_shape(self, data_path):
        with h5py.File(data_path) as h5file:
            in_features = h5file["train_encod_data"].shape[-1]
        return in_features, self.out_features

    def _get_state_dict(self, data_path):
        with h5py.File(data_path) as h5file:
            weight = h5file["readin_weight"][()]
            # bias = -np.dot(h5file["readout_bias"][()], weight)
            bias = h5file["readin_bias"][()]
        return {"weight": torch.tensor(weight.T), "bias": torch.tensor(bias)}


class MultisessionReadout(_MultisessionModuleList):
    def __init__(
        self,
        datafile_pattern: str,
        in_features: int = None,
        pcr_init: bool = True,
        requires_grad: bool = True,
        module: str = 'linear',
    ):
        assert (
            in_features is not None
        ) != pcr_init, "Setting `in_features` mutually excludes `pcr_init`."
        self.in_features = in_features
        super().__init__(
            datafile_pattern=datafile_pattern,
            pcr_init=pcr_init,
            requires_grad=requires_grad,
            module=module,
        )

    def _get_layer_shape(self, data_path):
        with h5py.File(data_path) as h5file:
            out_features = h5file["train_recon_data"].shape[-1]
        return self.in_features, out_features

    def _get_state_dict(self, data_path):
        with h5py.File(data_path) as h5file:
            weight = np.linalg.pinv(h5file["readin_weight"][()])
            bias = h5file["readout_bias"][()]
        return {"weight": torch.tensor(weight.T), "bias": torch.tensor(bias)}


class MultiCompartmentReadoutParametrized(nn.Module):
    def __init__(
        self,
        data_path: str,
        in_features: int,
    ):
        super().__init__()
        self.in_features = in_features
        self.layers = nn.ModuleList()
        
        # Load output dimensions for each area from the HDF5 file.
        with h5py.File(data_path, 'r') as h5file:
            out_features = h5file[f"train_recon_data"].shape[-1]
            # Choose the layer based on the parameterization flag.

            layer = InjectiveFlowReadout(
                in_features=in_features,
                out_features=out_features,
                hidden_size=256,
                num_layers=2,
                num_steps=6,
            )
            
            self.layers.append(layer)
                
        # Ensure gradients are enabled for all layers.
        for layer in self.layers:
            layer.requires_grad_(True)
    
    def forward(self, x):
        return self.layers[0](x)
class StiefelLinear(nn.Module):
    """
    Column-orthogonal (Stiefel) linear embedding:
      - Weight W ∈ R^{out_features × in_features} with W^T W = I_{in_features}
      - Left-inverse exists: z = W^T (y - b)

    Args:
        in_features  (int): input dimensionality D
        out_features (int): output dimensionality N (must satisfy N ≥ D)
        use_bias     (bool): include an additive bias term

    Methods:
        forward(x):        x[*, D] -> y[*, N]
        left_inverse(y):   y[*, N] -> x[*, D]   (exact, if this layer is the only mapping)
        weight:            returns the constrained weight tensor W
    """
    def __init__(self, in_features: int, out_features: int, use_bias: bool = True):
        super().__init__()
        assert out_features >= in_features, "StiefelLinear requires out_features >= in_features"
        self.linear = nn.Linear(in_features, out_features, bias=use_bias)

        # Constrain weight to the Stiefel manifold: W^T W = I_D (column-orthogonal)
        geotorch.orthogonal(self.linear, "weight")

        # (Optional but nice) initialize close to orthogonal
        with torch.no_grad():
            nn.init.orthogonal_(self.linear.weight)

    @property
    def weight(self) -> torch.Tensor:
        """Exposes the (constrained) weight W ∈ R^{N×D}."""
        return self.linear.weight

    @property
    def bias(self) -> Optional[torch.Tensor]:
        return self.linear.bias

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply y = W x + b. Shapes: x[..., D] -> y[..., N]."""
        return self.linear(x)

    @torch.no_grad()
    def left_inverse(self, y: torch.Tensor) -> torch.Tensor:
        """
        Exact left-inverse for this layer alone:
            x = W^T (y - b)
        Shapes: y[..., N] -> x[..., D]
        """
        if self.linear.bias is not None:
            y = y - self.linear.bias
        return y @ self.weight.t()
    

