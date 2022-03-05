import torch
import torch.nn.functional as F
from torch.distributions import Bernoulli


def pad_mask(mask, data, value):
    """Adds padding to I/O masks for CD and SV in cases where
    reconstructed data is not the same shape as the input data.
    """
    t_forward = data.shape[1] - mask.shape[1]
    n_heldout = data.shape[2] - mask.shape[2]
    pad_shape = (0, n_heldout, 0, t_forward)
    return F.pad(mask, pad_shape, value=value)


class CoordinatedDropout:
    def __init__(self, cd_rate, cd_pass_rate, ic_enc_seq_len):
        self.cd_rate = cd_rate
        self.ic_enc_seq_len = ic_enc_seq_len
        self.cd_input_dist = Bernoulli(1 - cd_rate)
        self.cd_pass_dist = Bernoulli(cd_pass_rate)

    def process_inputs(self, input_data):
        # Only use CD where we are inferring rates (none inferred for IC segment)
        unmaskable_data = input_data[:, : self.ic_enc_seq_len, :]
        maskable_data = input_data[:, self.ic_enc_seq_len :, :]
        # Sample a new CD mask at each training step
        device = input_data.device
        cd_mask = self.cd_input_dist.sample(maskable_data.shape).to(device)
        pass_mask = self.cd_pass_dist.sample(maskable_data.shape).to(device)
        # Save the gradient mask for `process_outputs`
        if self.cd_rate > 0:
            self.grad_mask = torch.logical_or(
                torch.logical_not(cd_mask), pass_mask
            ).float()
        else:
            # If cd_rate == 0, turn off CD
            self.grad_mask = torch.ones_like(cd_mask)
        # Mask and scale post-CD input so it has the same sum as the original data
        cd_masked_data = maskable_data * cd_mask / (1 - self.cd_rate)
        # Concatenate the data from the IC encoder segment if using
        cd_input = torch.cat([unmaskable_data, cd_masked_data], axis=1)

        return cd_input

    def process_outputs(self, output_data):
        # Expand mask, but don't block gradients
        grad_mask = pad_mask(self.grad_mask, output_data, 1.0)
        # Block gradients with respect to the masked outputs
        grad_data = output_data * grad_mask
        nograd_data = (output_data * (1 - grad_mask)).detach()
        cd_output = grad_data + nograd_data

        return cd_output


class SampleValidation:
    def __init__(self, sv_rate, ic_enc_seq_len, recon_reduce_mean):
        self.sv_rate = sv_rate
        self.ic_enc_seq_len = ic_enc_seq_len
        self.recon_reduce_mean = recon_reduce_mean

    def process_inputs(self, input_data, sv_mask):
        # Only use SV where we are inferring rates (none inferred for IC segment)
        unmaskable_data = input_data[:, : self.ic_enc_seq_len, :]
        maskable_data = input_data[:, self.ic_enc_seq_len :, :]
        # Set heldout data to zero and scale up heldin data
        sv_masked_data = maskable_data * sv_mask / (1 - self.sv_rate)
        # Concatenate the data from the IC encoder segment if using
        sv_input = torch.cat([unmaskable_data, sv_masked_data], axis=1)

        return sv_input

    def process_outputs(self, recon_loss, sv_mask, log_fn, data_split):
        # Aggregate and log recon cost for samples heldout for SV
        if self.sv_rate == 0:
            # Skip the masking if SV is not being used
            recon_heldin = recon_loss
            recon_heldout_agg = torch.tensor(float("nan"))
        else:
            # Rescale so means are comparable
            heldin_mask = sv_mask / (1 - self.sv_rate)
            heldout_mask = (1 - sv_mask) / self.sv_rate
            # Apply the heldin mask - expand the mask as necessary
            heldin_mask = pad_mask(heldin_mask, recon_loss, value=1.0)
            recon_heldin = recon_loss * heldin_mask
            # Apply the heldout mask - only include points with encoder input
            _, t_enc, n_enc = heldout_mask.shape
            encod_recon_loss = recon_loss[:, :t_enc, :n_enc]
            recon_heldout_masked = encod_recon_loss * heldout_mask
            # Aggregate the heldout cost for logging
            if not self.recon_reduce_mean:
                recon_heldout_masked = torch.sum(recon_heldout_masked, dim=(1, 2))
            recon_heldout_agg = torch.mean(recon_heldout_masked)
        # Log the heldout reconstruction cost
        log_fn(f"{data_split}/recon/sv", recon_heldout_agg)

        return recon_heldin
