from typing import Optional

import h5py
import pytorch_lightning as pl
import torch
from torch.distributions import Bernoulli
from torch.utils.data import DataLoader, TensorDataset


def to_tensor(array):
    return torch.tensor(array, dtype=torch.float)


def add_auxiliary_data(datamodule, data_dict={}):
    hps = datamodule.hparams
    # Unpack the data tuples
    train_encod_data, train_recon_data, *train_other = datamodule.train_data
    valid_encod_data, valid_recon_data, *valid_other = datamodule.valid_data
    # Create sample validation masks
    torch.manual_seed(hps.dm_sv_seed)
    sv_input_dist = Bernoulli(1 - hps.dm_sv_rate)
    train_sv_mask = sv_input_dist.sample(train_encod_data.shape)
    valid_sv_mask = sv_input_dist.sample(valid_encod_data.shape)
    # Remove sample validation during the IC encoder segment
    train_sv_mask = train_sv_mask[:, hps.dm_ic_enc_seq_len :, :]
    valid_sv_mask = valid_sv_mask[:, hps.dm_ic_enc_seq_len :, :]
    # Load or simulate external inputs
    if "train_ext_input" in data_dict:
        train_ext = to_tensor(data_dict["train_ext_input"])
        valid_ext = to_tensor(data_dict["valid_ext_input"])
    else:
        train_ext = torch.zeros_like(train_encod_data[..., :0])
        valid_ext = torch.zeros_like(valid_encod_data[..., :0])
    # Remove external inputs during the IC encoder segment
    train_ext = train_ext[:, hps.dm_ic_enc_seq_len :, :]
    valid_ext = valid_ext[:, hps.dm_ic_enc_seq_len :, :]
    # Load or simulate ground truth
    if "train_truth" in data_dict:
        cf = data_dict["conversion_factor"]
        train_truth = to_tensor(data_dict["train_truth"]) / cf
        valid_truth = to_tensor(data_dict["valid_truth"]) / cf
    else:
        train_truth = torch.full_like(train_recon_data, float("nan"))
        valid_truth = torch.full_like(valid_recon_data, float("nan"))
    # Remove ground truth during the IC encoder segment
    train_truth = train_truth[:, hps.dm_ic_enc_seq_len :, :]
    valid_truth = valid_truth[:, hps.dm_ic_enc_seq_len :, :]
    # Overwrite the data tuples and datasets
    datamodule.train_data = (
        train_encod_data,
        train_recon_data,
        train_sv_mask,
        train_ext,
        train_truth,
        *train_other,
    )
    datamodule.valid_data = (
        valid_encod_data,
        valid_recon_data,
        valid_sv_mask,
        valid_ext,
        valid_truth,
        *valid_other,
    )
    datamodule.train_ds = TensorDataset(*datamodule.train_data)
    datamodule.valid_ds = TensorDataset(*datamodule.valid_data)


class BasicDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_path: str,
        batch_size: int = 64,
        num_workers: int = 4,
        dm_sv_rate: float = 0.0,
        dm_sv_seed: int = 0,
        dm_ic_enc_seq_len: int = 0,
    ):
        super().__init__()
        self.save_hyperparameters()

    def setup(self, stage: Optional[str] = None):
        hps = self.hparams
        # Load data arrays from the file
        with h5py.File(hps.data_path, "r") as h5file:
            data_dict = {k: v[()] for k, v in h5file.items()}
        # Add the data to be modeled
        train_data = to_tensor(data_dict["train_data"])
        valid_data = to_tensor(data_dict["valid_data"])
        # Create and store the datasets
        self.train_data = (train_data, train_data)
        self.valid_data = (valid_data, valid_data)
        # Add auxiliary data for LFADS
        add_auxiliary_data(self, data_dict)

    def train_dataloader(self, shuffle=True):
        train_dl = DataLoader(
            self.train_ds,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            shuffle=shuffle,
        )
        return train_dl

    def val_dataloader(self):
        valid_dl = DataLoader(
            self.valid_ds,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
        )
        return valid_dl
