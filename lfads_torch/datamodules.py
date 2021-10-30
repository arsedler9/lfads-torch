from typing import Optional

import h5py
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, TensorDataset


def to_tensor(array):
    return torch.tensor(array, dtype=torch.float)


class HDF5DataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_path: str,
        batch_size: int = 64,
        num_workers: int = 0,
        ext_input: bool = False,
        ground_truth: bool = False,
    ):
        super().__init__()
        self.save_hyperparameters()

    def setup(self, stage: Optional[str] = None):
        hps = self.hparams
        # Load data arrays from the file
        with h5py.File(hps.data_path, "r") as h5file:
            data_dict = {k: v[()] for k, v in h5file.items()}
        # Add the data to be modeled
        train_tensors = [to_tensor(data_dict["train_data"])]
        valid_tensors = [to_tensor(data_dict["valid_data"])]
        if hps.ext_input:
            # Add the external inputs
            train_tensors.append(to_tensor(data_dict["train_ext_input"]))
            valid_tensors.append(to_tensor(data_dict["valid_ext_input"]))
        if hps.ground_truth:
            # Add the ground truth parameters
            train_tensors.append(to_tensor(data_dict["train_truth"]))
            valid_tensors.append(to_tensor(data_dict["valid_truth"]))
        # Create and store the datasets
        self.train_ds = TensorDataset(*train_tensors)
        self.valid_ds = TensorDataset(*valid_tensors)

    def train_dataloader(self):
        train_dl = DataLoader(
            self.train_ds,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            shuffle=True,
        )
        return train_dl

    def val_dataloader(self):
        valid_dl = DataLoader(
            self.valid_ds,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
        )
        return valid_dl
