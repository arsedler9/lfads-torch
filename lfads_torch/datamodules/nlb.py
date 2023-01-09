# Must have `nlb_lightning` installed
import torch
import torch.nn.functional as F
from nlb_lightning.datamodules import NLBDataModule
from pytorch_lightning.trainer.supporters import CombinedLoader
from torch.utils.data import DataLoader

from .base import attach_tensors, reshuffle_train_valid


class LFADSNLBDataModule(NLBDataModule):
    def __init__(
        self,
        dataset_name: str = "mc_maze_large",
        phase: str = "val",
        bin_width: int = 5,
        batch_size: int = 64,
        reshuffle_tv_seed: int = None,
        num_workers: int = 4,
        sv_rate: float = 0.0,
        sv_seed: int = 0,
        dm_ic_enc_seq_len: int = 0,
    ):
        """Extends the `NLBDataModule` for LFADS.

        Parameters
        ----------
        dataset_name : str, optional
            One of the data tags specified by the NLB organizers,
            by default "mc_maze_large"
        phase : str, optional
            The phase of the competition - either "val" or "test",
            by default "val"
        bin_width : int, optional
            The width of data bins, by default 5
        batch_size : int, optional
            The number of samples to process in each batch,
            by default 64
        num_workers : int, optional
            The number of subprocesses to use for data loading,
            by default 4
        sv_rate : float, optional
            The fraction of data elements to use for sample
            validation, by default 0.0
        sv_seed : int, optional
            The seed to use for sample validation masks,
            by default 0
        dm_ic_enc_seq_len : int, optional
            The number of time steps to use solely for encoding
            the initial condition, by default 0
        """
        super().__init__(
            dataset_name=dataset_name,
            phase=phase,
            bin_width=bin_width,
            batch_size=batch_size,
            num_workers=num_workers,
        )
        self.save_hyperparameters()

    def setup(self, stage=None):
        # Load the data tuples as defined in nlb_lightning
        super().setup(stage=stage)
        hps = self.hparams
        if hps.reshuffle_tv_seed is not None:
            # Reshuffle the training / validation split
            self.train_data, self.valid_data = reshuffle_train_valid(
                train_tensors=self.train_data,
                valid_tensors=self.valid_data,
                seed=hps.reshuffle_tv_seed,
            )
        train_encod_data, train_recon_data, train_behavior = self.train_data
        if hps.phase == "test":
            # Generate placeholders for the missing test-phase data
            (valid_encod_data,) = self.valid_data
            behavior_shape = (len(valid_encod_data),) + train_behavior.shape[1:]
            valid_behavior = torch.full(behavior_shape, float("nan"))
            t_forward = train_recon_data.shape[1] - valid_encod_data.shape[1]
            n_heldout = train_recon_data.shape[2] - valid_encod_data.shape[2]
            pad_shape = (0, n_heldout, 0, t_forward)
            valid_recon_data = F.pad(valid_encod_data, pad_shape, value=float("nan"))
        else:
            valid_encod_data, valid_recon_data, valid_behavior = self.valid_data
        data_dicts = [
            {
                "train_encod_data": train_encod_data.numpy(),
                "train_recon_data": train_recon_data.numpy(),
                "train_behavior": train_behavior.numpy(),
                "valid_encod_data": valid_encod_data.numpy(),
                "valid_recon_data": valid_recon_data.numpy(),
                "valid_behavior": valid_behavior.numpy(),
            }
        ]
        # Add auxiliary data for LFADS
        attach_tensors(self, data_dicts, extra_keys=["behavior"])

    def train_dataloader(self, shuffle=True):
        # PTL provides all batches at once, so divide amongst dataloaders
        batch_size = int(self.hparams.batch_size / len(self.train_ds))
        dataloaders = {
            i: DataLoader(
                ds,
                batch_size=batch_size,
                num_workers=self.hparams.num_workers,
                shuffle=shuffle,
            )
            for i, ds in enumerate(self.train_ds)
        }
        return CombinedLoader(dataloaders, mode="max_size_cycle")

    def val_dataloader(self):
        # PTL provides all batches at once, so divide amongst dataloaders
        batch_size = int(self.hparams.batch_size / len(self.train_ds))
        dataloaders = {
            i: DataLoader(
                ds,
                batch_size=batch_size,
                num_workers=self.hparams.num_workers,
            )
            for i, ds in enumerate(self.valid_ds)
        }
        return CombinedLoader(dataloaders, mode="max_size_cycle")

    def predict_dataloader(self):
        # NOTE: Returning dicts of DataLoaders is incompatible with trainer.predict,
        # but convenient for posterior sampling. Can't use CombinedLoader here because
        # we only want to see each sample once.
        dataloaders = {
            s: {
                "train": DataLoader(
                    self.train_ds[s],
                    batch_size=self.hparams.batch_size,
                    num_workers=self.hparams.num_workers,
                    shuffle=False,
                ),
                "valid": DataLoader(
                    self.valid_ds[s],
                    batch_size=self.hparams.batch_size,
                    num_workers=self.hparams.num_workers,
                    shuffle=False,
                ),
            }
            for s in range(len(self.train_ds))
        }
        return dataloaders
