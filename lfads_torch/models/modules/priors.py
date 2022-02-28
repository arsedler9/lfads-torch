import torch
from torch import nn
from torch.distributions import Independent, Normal, kl_divergence


class MultivariateNormal(nn.Module):
    def __init__(
        self,
        mean,
        variance,
        shape,
        trainable_means=True,
        trainable_vars=False,
    ):
        super().__init__()
        # Create distribution parameter tensors
        means = torch.ones(shape) * mean
        logvars = torch.log(torch.ones(shape) * variance)
        self.mean = nn.Parameter(means, requires_grad=trainable_means)
        self.logvar = nn.Parameter(logvars, requires_grad=trainable_vars)

    def forward(self, post_mean, post_std):
        # Create the posterior distribution
        posterior = Independent(Normal(post_mean, post_std), 1)
        # Create the prior and posterior
        prior_std = torch.exp(0.5 * self.logvar)
        prior = Independent(Normal(self.mean, prior_std), 1)
        # Compute KL analytically
        kl_batch = kl_divergence(posterior, prior)
        return torch.mean(kl_batch)
