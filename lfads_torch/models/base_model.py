import pytorch_lightning as pl
import torch
from torch import nn
from torch.distributions import Independent, Normal

from ..metrics import r2_score
from .modules import recons
from .modules.decoder import Decoder
from .modules.encoder import Encoder
from .modules.initializers import init_variance_scaling_
from .modules.l2 import compute_l2_penalty
from .modules.overfitting import CoordinatedDropout, SampleValidation

# def validate_hparams(hparams: dict):
#     hps = dotdict(hparams)

#     # Check seq_len and ic_enc_seq_len logic
#     assert hps.seq_len > hps.ic_enc_seq_len
#     hparams['ic_enc_seq_len'] = max(hps.ic_enc_seq_len, 0)
#     if hps.ic_enc_seq_len > 0:
#         print(f"Using the first {hps.ic_enc_seq_len} steps "
#         "to encode initial condition. Inferring rates for the "
#         f"remaining {hps.seq_len - hps.ic_enc_seq_len} steps.")
#     # Decide whether to use the controller

#     assert hparams['ic_enc_seq_len']


class LFADS(pl.LightningModule):
    def __init__(
        self,
        data_dim: int,
        ext_input_dim: int,
        seq_len: int,
        ic_enc_seq_len: int,
        ic_enc_dim: int,
        ci_enc_dim: int,
        ci_lag: int,
        con_dim: int,
        co_dim: int,
        ic_dim: int,
        gen_dim: int,
        fac_dim: int,
        dropout_rate: float,
        cd_rate: float,
        cd_pass_rate: float,
        sv_rate: float,
        sv_seed: int,
        reconstruction: recons.Reconstruction,
        sample_posteriors: bool,
        co_prior: nn.Module,
        ic_prior: nn.Module,
        ic_post_var_min: float,
        cell_clip: float,
        loss_scale: float,
        recon_reduce_mean: bool,
        lr_init: float,
        lr_stop: float,
        lr_decay: float,
        lr_patience: int,
        lr_adam_epsilon: float,
        l2_start_epoch: int,
        l2_increase_epoch: int,
        l2_ic_enc_scale: float,
        l2_ci_enc_scale: float,
        l2_gen_scale: float,
        l2_con_scale: float,
        kl_start_epoch: int,
        kl_increase_epoch: int,
        kl_ic_scale: float,
        kl_co_scale: float,
    ):
        super().__init__()
        self.save_hyperparameters()

        # Decide whether to use the controller
        self.use_con = all([ci_enc_dim > 0, con_dim > 0, co_dim > 0])
        # Create the encoder and decoder
        self.encoder = Encoder(hparams=self.hparams)
        self.decoder = Decoder(hparams=self.hparams)
        # Create object to manage coordinated dropout
        self.coord_dropout = CoordinatedDropout(cd_rate, cd_pass_rate, ic_enc_seq_len)
        # Create object to manage sample validation
        self.samp_validation = SampleValidation(
            sv_rate, ic_enc_seq_len, recon_reduce_mean
        )
        # Create object to manage reconstruction
        self.recon = reconstruction
        # Create the mapping from factors to output parameters
        self.output_linear = nn.Linear(fac_dim, data_dim * self.recon.n_params)
        init_variance_scaling_(self.output_linear.weight, fac_dim)
        # Store the trainable priors
        self.ic_prior = ic_prior
        if self.use_con:
            self.co_prior = co_prior

    def forward(self, data, ext_input, output_means=True):
        hps = self.hparams
        # Pass the data through the encoders
        ic_mean, ic_std, ci = self.encoder(data)
        # Create the posterior distribution over initial conditions
        ic_post = Independent(Normal(ic_mean, ic_std), 1)
        # Choose to take a sample or to pass the mean
        ic_samp = ic_post.rsample() if hps.sample_posteriors else ic_mean
        # Unroll the decoder to estimate latent states
        (
            gen_init,
            gen_states,
            con_states,
            co_means,
            co_stds,
            gen_inputs,
            factors,
        ) = self.decoder(ic_samp, ci, ext_input)
        # Convert the factors representation into output distribution parameters
        output_params = self.output_linear(factors)
        # Separate parameters of the output distribution
        output_params = self.recon.reshape_output_params(output_params)
        # Convert the output parameters to means if requested
        if output_means:
            output_params = self.recon.compute_means(output_params)
        # Return the parameter estimates and all intermediate activations
        return (
            output_params,
            ic_mean,
            ic_std,
            co_means,
            co_stds,
            factors,
            gen_states,
            gen_init,
            gen_inputs,
            con_states,
        )

    def configure_optimizers(self):
        hps = self.hparams
        # Create an optimizer
        optimizer = torch.optim.Adam(self.parameters(), lr=hps.lr_init)
        # Create a scheduler to reduce the learning rate over time
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer=optimizer,
            mode="min",
            factor=hps.lr_decay,
            patience=hps.lr_patience,
            threshold=0.0,
            min_lr=hps.lr_stop,
            eps=hps.lr_adam_epsilon,
            verbose=True,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler,
            "monitor": "valid/recon",
        }

    def training_step(self, batch, batch_ix):
        hps = self.hparams
        data, sv_mask, ext_input, truth = batch
        # Apply sample validation processing to the input data
        sv_data = self.samp_validation.process_inputs(data, sv_mask)
        # Apply coordinated dropout processing to the input data
        cd_data = self.coord_dropout.process_inputs(sv_data)
        # Perform the forward pass
        output_params, ic_mean, ic_std, co_means, co_stds, *_ = self.forward(
            cd_data,
            ext_input,
            output_means=False,
        )
        # Compute the reconstruction loss
        recon_all = self.recon.compute_loss(data, output_params)
        # Apply coordinated dropout processing to the recon costs
        recon_all = self.coord_dropout.process_outputs(recon_all)
        # Apply sample validation processing to the recon costs
        recon_all = self.samp_validation.process_outputs(
            recon_all, sv_mask, self.log, "train"
        )
        # Aggregate the heldout cost for logging
        if not hps.recon_reduce_mean:
            recon_all = torch.sum(recon_all, dim=(1, 2))
        recon = torch.mean(recon_all)
        # Compute the L2 penalty on recurrent weights
        l2 = compute_l2_penalty(self, self.hparams)
        l2_ramp = (self.current_epoch - hps.l2_start_epoch) / (
            hps.l2_increase_epoch + 1
        )
        # Compute the KL penalty on posteriors
        ic_kl = self.ic_prior(ic_mean, ic_std) * self.hparams.kl_ic_scale
        co_kl = self.co_prior(co_means, co_stds) * self.hparams.kl_co_scale
        kl_ramp = (self.current_epoch - hps.kl_start_epoch) / (
            hps.kl_increase_epoch + 1
        )
        # Clamp the ramps
        l2_ramp = torch.clamp(torch.tensor(l2_ramp), 0, 1)
        kl_ramp = torch.clamp(torch.tensor(kl_ramp), 0, 1)
        # Compute the final loss
        loss = hps.loss_scale * (recon + l2_ramp * l2 + kl_ramp * (ic_kl + co_kl))
        # Compute the reconstruction accuracy, if applicable
        r2 = r2_score(self.recon.compute_means(output_params), truth)
        # Log all of the metrics
        metrics = {
            "train/loss": loss,
            "train/recon": recon,
            "train/r2": r2,
            "train/wt_l2": l2,
            "train/wt_l2/ramp": l2_ramp,
            "train/wt_kl": ic_kl + co_kl,
            "train/wt_kl/ic": ic_kl,
            "train/wt_kl/co": co_kl,
            "train/wt_kl/ramp": kl_ramp,
        }
        self.log_dict(metrics)

        return loss

    def validation_step(self, batch, batch_ix):
        hps = self.hparams
        data, sv_mask, ext_input, truth = batch
        # Apply sample validation processing to the input data
        sv_data = self.samp_validation.process_inputs(data, sv_mask)
        # Perform the forward pass
        output_params, ic_mean, ic_std, co_means, co_stds, *_ = self.forward(
            sv_data,
            ext_input,
            output_means=False,
        )
        # Compute the reconstruction loss
        recon_all = self.recon.compute_loss(data, output_params)
        # Apply sample validation processing to the recon costs
        recon_all = self.samp_validation.process_outputs(
            recon_all, sv_mask, self.log, "valid"
        )
        # Aggregate the heldout cost for logging
        if not hps.recon_reduce_mean:
            recon_all = torch.sum(recon_all, dim=(1, 2))
        recon = torch.mean(recon_all)
        # Compute the L2 penalty on recurrent weights
        l2 = compute_l2_penalty(self, self.hparams)
        l2_ramp = (self.current_epoch - hps.l2_start_epoch) / (
            hps.l2_increase_epoch + 1
        )
        # Compute the KL penalty on posteriors
        ic_kl = self.ic_prior(ic_mean, ic_std) * self.hparams.kl_ic_scale
        co_kl = self.co_prior(co_means, co_stds) * self.hparams.kl_co_scale
        kl_ramp = (self.current_epoch - hps.kl_start_epoch) / (
            hps.kl_increase_epoch + 1
        )
        # Clamp the ramps
        l2_ramp = torch.clamp(torch.tensor(l2_ramp), 0, 1)
        kl_ramp = torch.clamp(torch.tensor(kl_ramp), 0, 1)
        # Compute the final loss
        loss = hps.loss_scale * (recon + l2_ramp * l2 + kl_ramp * (ic_kl + co_kl))
        # Compute the reconstruction accuracy, if applicable
        r2 = r2_score(self.recon.compute_means(output_params), truth)
        # Log all of the metrics
        metrics = {
            "valid/loss": loss,
            "valid/recon": recon,
            "valid/r2": r2,
            "valid/wt_l2": l2,
            "valid/wt_l2/ramp": l2_ramp,
            "valid/wt_kl": ic_kl + co_kl,
            "valid/wt_kl/ic": ic_kl,
            "valid/wt_kl/co": co_kl,
            "valid/wt_kl/ramp": kl_ramp,
            "hp_metric": recon,
            "cur_epoch": float(self.current_epoch),
        }
        self.log_dict(metrics)

        return loss

    def update_hparams(self, hparams):
        raise NotImplementedError
