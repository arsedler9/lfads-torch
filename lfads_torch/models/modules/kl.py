import torch
from torch.distributions import Independent, Normal, kl_divergence


def make_posteriors(ic_mean, ic_std, co_means, co_stds):
    ic_post = Independent(Normal(ic_mean, ic_std), 1)
    co_post = Independent(Normal(co_means, co_stds), 1)

    return ic_post, co_post


def compute_kl_penalties(lfads, ic_mean, ic_std, co_means, co_stds):
    ic_post, co_post = make_posteriors(ic_mean, ic_std, co_means, co_stds)
    # Create the IC priors
    ic_prior_std = torch.exp(0.5 * lfads.ic_prior_logvar)
    ic_prior = Independent(Normal(lfads.ic_prior_mean, ic_prior_std), 1)
    # Compute KL for the IC's analytically
    ic_kl_batch = kl_divergence(ic_post, ic_prior)
    wt_ic_kl = torch.mean(ic_kl_batch) * lfads.hparams.kl_ic_weight
    # Create the CO priors
    co_prior_std = torch.exp(0.5 * lfads.co_prior_logvar)
    co_prior = Independent(Normal(lfads.co_prior_mean, co_prior_std), 1)
    # Compute KL for CO's analytially, average across time and batch
    co_kl_batch = kl_divergence(co_post, co_prior)
    wt_co_kl = torch.mean(co_kl_batch) * lfads.hparams.kl_co_weight

    return wt_ic_kl, wt_co_kl
