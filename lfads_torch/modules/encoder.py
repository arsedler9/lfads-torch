import torch
from torch import nn
from torch.distributions import Independent, Normal

from .recurrent import BidirectionalClippedGRU


class dotdict(dict):
    """Access dictionary attributes with dot notation"""

    __getitem__ = dict.get


class Encoder(nn.Module):
    def __init__(self, hparams: dict):
        super().__init__()
        hps = dotdict(hparams)
        self.hparams = hps

        # Initial hidden state for IC encoder
        self.ic_enc_h0 = torch.zeros((2, 1, hps.ic_enc_dim), requires_grad=True)
        # Initial condition encoder
        self.ic_enc = BidirectionalClippedGRU(
            input_size=hps.data_dim,
            hidden_size=hps.ic_enc_dim,
            clip_value=hps.clip_value,
        )
        # Mapping from final IC encoder state to IC parameters
        self.ic_linear = nn.Linear(hps.ic_enc_dim * 2, hps.ic_dim * 2)
        nn.init.normal_(self.ic_linear.weight, std=1 / torch.sqrt(hps.ic_enc_dim * 2))
        # Decide whether to use the controller
        self.use_con = all(
            [
                hps.ci_enc_dim > 0,
                hps.con_dim > 0,
                hps.co_dim > 0,
            ]
        )
        if self.use_con:
            # Initial hidden state for CI encoder
            self.ci_enc_h0 = torch.zeros((2, 1, hps.ci_enc_dim), requires_grad=True)
            # CI encoder
            self.ci_enc = BidirectionalClippedGRU(
                input_size=hps.data_dim,
                hidden_size=hps.ci_enc_dim,
                clip_value=hps.clip_value,
            )
        # Activation dropout layer
        self.dropout = nn.Dropout(hps.dropout_rate)

    def forward(self, data: torch.Tensor):
        hps = self.hparams

        # Pass data through IC encoder
        data_drop = self.dropout(data)
        _, h_n = self.ic_enc(data_drop, self.ic_enc_h0)
        # Combine output from fwd and bwd encoders
        h_n = torch.cat([*h_n], -1)
        # Compute initial condition posterior
        h_n_drop = self.dropout(h_n)
        ic_params = self.ic_linear(h_n_drop)
        ic_mean, ic_logstd = torch.split(ic_params, hps.ic_dim, dim=1)
        ic_post = Independent(Normal(ic_mean, torch.exp(ic_logstd)))
        # Pass data through CI encoder
        ci, _ = self.ci_enc(data_drop, self.ci_enc_h0)
        # Add a lag to the controller input
        ci_fwd, ci_bwd = torch.split(ci, hps.ci_enc_dim, axis=2)
        ci_fwd = nn.functional.pad(ci_fwd, (0, 0, hps.ci_lag, 0, 0, 0))
        ci_bwd = nn.functional.pad(ci_bwd, (0, 0, 0, hps.ci_lag, 0, 0))
        ci = torch.cat([ci_fwd[:, : hps.seq_len, :], ci_bwd[:, -hps.seq_len :, :]])
        return ic_post, ci

    def update_hparams(self, hparams: dict):
        pass
