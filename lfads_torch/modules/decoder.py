import torch
import torch.nn.functional as F
from torch import nn
from torch.distributions import Independent, Normal

from ..utils import dotdict
from .recurrent import ClippedGRUCell


class KernelNormalizedLinear(nn.Linear):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.unnormed_weight = self.weight

    def forward(self, input):
        self.weight = F.normalize(self.unnormed_weight, p=2, dim=0)
        return super().forward(input)


class DecoderCell(nn.Module):
    def __init__(self, hparams: dict):
        super().__init__()
        hps = dotdict(hparams)
        self.hparams = hps

        # Create the generator
        self.gen_cell = ClippedGRUCell(
            hps.ext_input_dim + hps.co_dim, hps.gen_dim, clip_value=hps.clip_value
        )
        # Create the mapping from generator states to factors
        self.fac_linear = KernelNormalizedLinear(hps.gen_dim, hps.fac_dim, bias=False)
        # Create the dropout layer
        self.dropout = nn.Dropout(hps.dropout_rate)
        # Decide whether to use the controller
        self.use_con = all(
            [
                hps.ci_enc_dim > 0,
                hps.con_dim > 0,
                hps.co_dim > 0,
            ]
        )
        if self.use_con:
            # Create the controller
            self.con_cell = ClippedGRUCell(
                hps.ci_dim + hps.fac_dim, hps.con_dim, clip_value=hps.clip_value
            )
            # Define the mapping from controller state to controller output parameters
            self.co_linear = nn.Linear(hps.con_dim, hps.co_dim * 2)
            nn.init.normal_(self.co_linear.weight, std=1 / torch.sqrt(hps.con_dim))
        # Keep track of the state dimensions
        self.state_dims = [
            hps.gen_dim,
            hps.con_dim,
            hps.co_dim,
            hps.co_dim,
            hps.co_dim,
            hps.fac_dim,
        ]
        # Keep track of the input dimensions
        self.input_dims = [hps.ci_enc_dim, hps.ext_input_dim]

    def forward(self, input: torch.Tensor, h_0: torch.Tensor):
        hps = self.hparams

        # Split the state up into variables of interest
        gen_state, con_state, co_mean, co_logstd, gen_input, factor = torch.split(
            h_0, self.state_dims, dim=1
        )
        ci_step, ext_input_step = torch.split(input, self.input_dims, axis=1)

        if self.use_con:
            # Compute controller inputs with dropout
            con_input = torch.cat([ci_step, factor], dim=1)
            con_input_drop = self.dropout(con_input)
            # Compute and store the next hidden state of the controller
            con_state = self.con_cell(con_input_drop, con_state)
            # Compute the distribution of the controller outputs at this timestep
            co_params = self.co_linear(con_state)
            co_mean, co_logstd = torch.split(co_params, hps.co_dim, dim=1)
            # Generate controller outputs
            if hps.sample_posteriors:
                # Sample from the distribution of controller outputs
                co_post = Independent(Normal(co_mean, torch.exp(co_logstd)))
                con_output = co_post.sample()
            else:
                # Pass mean in deterministic mode
                con_output = co_mean
            # Combine controller output with any external inputs
            gen_input = torch.cat([con_output, ext_input_step], dim=1)
        else:
            # If no controller is being used, can still provide ext inputs
            gen_input = ext_input_step
        # compute and store the next
        gen_state = self.gen_cell(gen_input, gen_state)
        gen_state_drop = self.dropout(gen_state)
        factor = self.fac_linear(gen_state_drop)

        hidden = torch.cat(
            [gen_state, con_state, co_mean, co_logstd, gen_input, factor]
        )

        return hidden


class DecoderRNN(nn.Module):
    def __init__(self, hparams: dict):
        super().__init__()
        self.cell = DecoderCell(hparams=hparams)

    def forward(self, input: torch.Tensor, h_0: torch.Tensor):
        hidden = h_0
        input = torch.transpose(input, 0, 1)
        output = []
        for input_step in input:
            hidden = self.cell(input_step, hidden)
            output.append(hidden)
        output = torch.stack(output, dim=1)
        return output, hidden
