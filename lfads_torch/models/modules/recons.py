"""This module specifies options for reconstruction losses and
loss-specific parameter processing.
Each loss class must set self.n_params for the number of parameters,
self.process_output_params which performs any fixed transformations
(may be different depending on boolean sample_and_average, i.e. rates
instead of logrates) and separates different parameters in a new inner
dimension, and self.compute_loss which computes the loss for given
tensors of data and inferred parameters.
"""

import abc

import torch
import torch.nn.functional as F


class Reconstruction(abc.ABC):
    @abc.abstractmethod
    def __init__(self):
        pass

    @abc.abstractmethod
    def reshape_output_params(self, output_params):
        pass

    @abc.abstractmethod
    def compute_loss(self, data, output_params):
        pass

    @abc.abstractmethod
    def compute_means(self, output_params):
        pass


class Poisson(Reconstruction):
    def __init__(self):
        self.n_params = 1

    def reshape_output_params(self, output_params):
        return torch.unsqueeze(output_params, dim=-1)

    def compute_loss(self, data, output_params):
        return F.poisson_nll_loss(
            output_params[..., 0],
            data,
            full=True,
            reduction="none",
        )

    def compute_means(self, output_params):
        return torch.exp(output_params[..., 0])


class MSE(Reconstruction):
    def __init__(self):
        self.n_params = 1

    def reshape_output_params(self, output_params):
        return torch.unsqueeze(output_params, dim=-1)

    def compute_loss(self, data, output_params):
        return (data - output_params[..., 0]) ** 2

    def compute_means(self, output_params):
        return output_params[..., 0]


class Gaussian(Reconstruction):
    def __init__(self):
        self.n_params = 2

    def reshape_output_params(self, output_params, sample_and_average):
        output_means, output_logvars = torch.split(output_params, 2, -1)
        return torch.stack([output_means, output_logvars], -1)

    def compute_loss(self, data, output_params):
        means, logvars = torch.unstack(output_params, axis=-1)
        recon_all = F.gaussian_nll_loss(
            input=means, target=data, var=torch.exp(logvars), reduction="none"
        )
        return recon_all

    def compute_means(self, output_params):
        return output_params[..., 0]
