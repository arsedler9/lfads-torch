import torch
import torch.nn.functional as F
from torch import nn
from torch.distributions import Independent, Normal

from ...utils import dotdict
from .recurrent import ClippedGRUCell


class KernelNormalizedLinear(nn.Linear):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.unnormed_weight = self.weight

    def forward(self, input):
        self.weight = F.normalize(self.unnormed_weight, p=2, dim=0)
        return super().forward(input)


class DecoderCell(nn.Module):
    def __init__(self, hparams):
        super().__init__()
        hps = dotdict(hparams)
        self.hparams = hps
        # Create the generator
        self.gen_cell = ClippedGRUCell(
            hps.ext_input_dim + hps.co_dim, hps.gen_dim, clip_value=hps.cell_clip
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
            # Initial hidden state for controller
            self.con_h0 = nn.Parameter(
                torch.zeros((1, hps.con_dim), requires_grad=True)
            )
            # Create the controller
            self.con_cell = ClippedGRUCell(
                hps.ci_enc_dim + hps.fac_dim, hps.con_dim, clip_value=hps.cell_clip
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

    def forward(self, input, h_0):
        hps = self.hparams

        # Split the state up into variables of interest
        gen_state, con_state, co_mean, co_std, gen_input, factor = torch.split(
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
            co_std = torch.exp(co_logstd)
            # Generate controller outputs
            if hps.sample_posteriors:
                # Sample from the distribution of controller outputs
                co_post = Independent(Normal(co_mean, co_std))
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

        hidden = torch.cat([gen_state, con_state, co_mean, co_std, gen_input, factor])

        return hidden


class DecoderRNN(nn.Module):
    def __init__(self, hparams):
        super().__init__()
        self.cell = DecoderCell(hparams=hparams)

    def forward(self, input, h_0):
        hidden = h_0
        input = torch.transpose(input, 0, 1)
        output = []
        for input_step in input:
            hidden = self.cell(input_step, hidden)
            output.append(hidden)
        output = torch.stack(output, dim=1)
        return output, hidden


class Decoder(nn.Module):
    def __init__(self, hparams):
        super().__init__()
        hps = dotdict(hparams)
        self.hparams = hps

        self.dropout = nn.Dropout(hps.dropout_rate)
        # Create the mapping from ICs to gen_state
        self.ic_to_g0 = nn.Linear(hps.ic_dim, hps.gen_dim)
        nn.init.normal_(self.ic_to_g0.weight, std=1 / torch.sqrt(hps.ic_dim))
        # Create the decoder RNN
        self.rnn = DecoderRNN(hparams=hparams)
        # Create the mapping from factors to rates
        self.lograte_linear = nn.Linear(hps.fac_dim, hps.data_dim)
        nn.init.normal_(self.lograte_linear.weight, std=1 / torch.sqrt(hps.fac_dim))

    def forward(self, ic_samp, ci, ext_input):
        hps = self.hparams

        # Get size of current batch (may be different than hps.batch_size)
        batch_size = ic_samp.shape[1]
        # Calculate initial generator state and pass it to the RNN with dropout rate
        gen_init = self.ic_to_g0(ic_samp)
        gen_init_drop = self.dropout(gen_init)
        # Perform dropout on the external inputs
        ext_input_drop = self.dropout(ext_input)
        # Prepare the decoder inputs and and initial state of decoder RNN
        dec_rnn_input = torch.cat([ci, ext_input_drop], axis=2)
        dec_rnn_h0 = torch.cat(
            [
                gen_init,
                self.rnn.cell.con_h0,
                torch.zeros((batch_size, hps.co_dim)),
                torch.zeros((batch_size, hps.co_dim)),
                torch.zeros((batch_size, hps.co_dim)),
                self.fac_linear(gen_init_drop),
            ],
            axis=1,
        )
        states = self.rnn(dec_rnn_input, dec_rnn_h0)
        split_states = torch.split(states, self.rnn.cell.state_dims, dim=1)
        dec_output = tuple(gen_init, *split_states)

        return dec_output
