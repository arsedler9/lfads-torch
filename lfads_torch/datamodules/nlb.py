# Must have `nlb_lightning` installed
from nlb_lightning.datamodules import NLBDataModule
from pytorch_lightning.trainer.supporters import CombinedLoader
from torch.utils.data import DataLoader

from .base import attach_tensors


class LFADSNLBDataModule(NLBDataModule):
    def __init__(
        self,
        dataset_name: str = "mc_maze_large",
        phase: str = "val",
        bin_width: int = 5,
        batch_size: int = 64,
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
        train_encod_data, train_recon_data, train_behavior = self.train_data
        # TODO: Update this it can handle test phase
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
        dataloaders = [
            DataLoader(
                ds,
                batch_size=batch_size,
                num_workers=self.hparams.num_workers,
                shuffle=shuffle,
            )
            for ds in self.train_ds
        ]
        return CombinedLoader(dataloaders, mode="max_size_cycle")

    def val_dataloader(self):
        # PTL provides all batches at once, so divide amongst dataloaders
        batch_size = int(self.hparams.batch_size / len(self.train_ds))
        dataloaders = [
            DataLoader(
                ds,
                batch_size=batch_size,
                num_workers=self.hparams.num_workers,
            )
            for ds in self.valid_ds
        ]
        return CombinedLoader(dataloaders, mode="max_size_cycle")
