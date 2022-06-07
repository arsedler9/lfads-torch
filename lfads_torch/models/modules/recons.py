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
from torch import nn


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

    def reshape_output_params(self, output_params):
        means, logvars = torch.chunk(output_params, 2, -1)
        return torch.stack([means, logvars], -1)

    def compute_loss(self, data, output_params):
        means, logvars = torch.unbind(output_params, axis=-1)
        recon_all = F.gaussian_nll_loss(
            input=means, target=data, var=torch.exp(logvars), reduction="none"
        )
        return recon_all

    def compute_means(self, output_params):
        return output_params[..., 0]


class Gamma(Reconstruction):
    def __init__(self):
        self.n_params = 2

    def reshape_output_params(self, output_params):
        logalphas, logbetas = torch.chunk(output_params, chunks=2, dim=-1)
        return torch.stack([logalphas, logbetas], -1)

    def compute_loss(self, data, output_params):
        alphas, betas = torch.unbind(torch.exp(output_params), axis=-1)
        output_dist = torch.distributions.Gamma(alphas, betas)
        recon_all = -output_dist.log_prob(data)
        return recon_all

    def compute_means(self, output_params):
        alphas, betas = torch.unbind(torch.exp(output_params), axis=-1)
        return alphas / betas


class ZeroInflatedGamma(nn.Module, Reconstruction):
    def __init__(
        self,
        recon_dim: int,
        gamma_loc: float,
        scale_init: float,
        scale_prior: float,
        scale_penalty: float,
    ):
        super().__init__()
        self.n_params = 3
        self.gamma_loc = gamma_loc
        # Initialize gamma parameter scaling weights
        scale_inits = torch.ones(2, recon_dim) * scale_init
        self.scale = nn.Parameter(scale_inits, requires_grad=True)
        self.scale_prior = scale_prior
        self.scale_penalty = scale_penalty

    def reshape_output_params(self, output_params):
        alpha_ps, beta_ps, q_ps = torch.chunk(output_params, chunks=3, dim=-1)
        return torch.stack([alpha_ps, beta_ps, q_ps], -1)

    def compute_loss(self, data, output_params):
        # Compute the scaled output parameters
        alphas, betas, qs = self._compute_scaled_params(output_params)
        # Shift data and replace zeros for convenient NLL calculation
        nz_ctr_data = torch.where(
            data == 0, torch.ones_like(data), data - self.gamma_loc
        )
        gamma = torch.distributions.Gamma(alphas, betas)
        recon_gamma = -gamma.log_prob(nz_ctr_data)
        # Replace with zero-inflated likelihoods
        recon_all = torch.where(
            data == 0, -torch.log(1 - qs), recon_gamma - torch.log(qs)
        )
        return recon_all

    def compute_means(self, output_params):
        # Compute the means of the ZIG distribution
        alphas, betas, qs = self._compute_scaled_params(output_params)
        return qs * (alphas / betas + self.gamma_loc)

    def compute_l2(self):
        # Compute an L2 scaling penalty on the gamma parameter scaling
        l2 = torch.sum((self.scale - self.scale_prior) ** 2)
        return 0.5 * self.scale_penalty * l2

    def _compute_scaled_params(self, output_params):
        # Compute sigmoid and clamp to avoid zero-valued rates
        sig_params = torch.clamp_min(torch.sigmoid(output_params), 1e-5)
        # Separate the parameters
        sig_alphas, sig_betas, qs = torch.unbind(sig_params, axis=-1)
        # Scale alphas and betas by per-neuron multiplicative factors
        alphas = sig_alphas * self.scale[0]
        betas = sig_betas * self.scale[1]
        return alphas, betas, qs
