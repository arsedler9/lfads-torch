"""This module specifies options for reconstruction losses and
loss-specific parameter processing.
Each loss class must set self.n_params for the number of parameters,
self.process_output_params which performs any fixed transformations
(may be different depending on boolean sample_and_average, i.e. rates
instead of logrates) and separates different parameters in a new inner
dimension, and self.compute_loss which computes the loss for given
tensors of data and inferred parameters.
"""

import torch
import torch.nn.functional as F


class Poisson:
    def __init__(self):
        self.n_params = 1

    def process_output_params(self, output_params, sample_and_average):
        if sample_and_average:
            output_params = torch.exp(output_params)
        output_params = torch.unsqueeze(output_params, dim=-1)
        return output_params

    def compute_loss(self, data, output_params):
        logrates = output_params[..., 0]
        recon_all = F.poisson_nll_loss(logrates, data, full=True, reduction="none")
        return recon_all

    def compute_mean(self, output_params):
        return torch.exp(output_params[..., 0])


class MSE:
    def __init__(self):
        self.n_params = 1

    def process_output_params(self, output_params, sample_and_average):
        output_params = torch.unsqueeze(output_params, dim=-1)
        return output_params

    def compute_loss(self, data, output_params):
        recon_data = output_params[..., 0]
        recon_all = (data - recon_data) ** 2
        return recon_all

    def compute_mean(self, output_params):
        return output_params[..., 0]


class Gaussian:
    def __init__(self):
        self.n_params = 2

    def process_output_params(self, output_params, sample_and_average):
        output_means, output_logvars = torch.split(output_params, 2, -1)
        output_vars = torch.exp(output_logvars)
        output_params = torch.stack([output_means, output_vars], -1)
        return output_params

    def compute_loss(self, data, output_params):
        means, vars = torch.unstack(output_params, axis=-1)
        recon_all = F.gaussian_nll_loss(
            input=means, target=data, var=vars, reduction="none"
        )
        return recon_all

    def compute_mean(self, output_params):
        return output_params[..., 0]
