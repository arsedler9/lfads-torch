import h5py
import pytorch_lightning as pl
import torch
from pytorch_lightning.trainer.supporters import CombinedLoader
from torch import Tensor
from torch.distributions import Bernoulli
from torch.utils.data import DataLoader, Dataset

from ..tuples import SessionBatch

MANDATORY_DATA_DICT_KEYS = [
    "train_encod_data",
    "train_recon_data",
    "valid_encod_data",
    "valid_recon_data",
]


def to_tensor(array):
    return torch.tensor(array, dtype=torch.float)


def attach_tensors(datamodule, data_dicts: list[dict], extra_keys: list[str] = []):
    hps = datamodule.hparams
    all_train_data, all_valid_data = [], []
    for data_dict in data_dicts:
        # Ensure that the data dict has all of the required keys
        assert all(key in data_dict for key in MANDATORY_DATA_DICT_KEYS)
        train_encod_data = to_tensor(data_dict["train_encod_data"])
        train_recon_data = to_tensor(data_dict["train_recon_data"])
        valid_encod_data = to_tensor(data_dict["valid_encod_data"])
        valid_recon_data = to_tensor(data_dict["valid_recon_data"])
        # Create sample validation masks
        torch.manual_seed(hps.sv_seed)
        sv_input_dist = Bernoulli(1 - hps.sv_rate)
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
        # Extract data for any extra keys
        train_other = [to_tensor(data_dict[f"train_{k}"]) for k in extra_keys]
        valid_other = [to_tensor(data_dict[f"valid_{k}"]) for k in extra_keys]
        # Store the data for this session
        all_train_data.append(
            (
                SessionBatch(
                    train_encod_data,
                    train_recon_data,
                    train_ext,
                    train_truth,
                    train_sv_mask,
                ),
                tuple(train_other),
            )
        )
        all_valid_data.append(
            (
                SessionBatch(
                    valid_encod_data,
                    valid_recon_data,
                    valid_ext,
                    valid_truth,
                    valid_sv_mask,
                ),
                tuple(valid_other),
            )
        )
    # Stack all of the data across sessions
    datamodule.train_data = all_train_data
    datamodule.valid_data = all_valid_data
    # Create a TensorDataset for each session
    datamodule.train_ds = [SessionDataset(*train_data) for train_data in all_train_data]
    datamodule.valid_ds = [SessionDataset(*valid_data) for valid_data in all_valid_data]


class SessionDataset(Dataset):
    def __init__(
        self, model_tensors: SessionBatch[Tensor], extra_tensors: tuple[Tensor]
    ):
        all_tensors = [*model_tensors, *extra_tensors]
        assert all(
            all_tensors[0].size(0) == tensor.size(0) for tensor in all_tensors
        ), "Size mismatch between tensors"
        self.model_tensors = model_tensors
        self.extra_tensors = extra_tensors

    def __getitem__(self, index):
        model_tensors = SessionBatch(*[t[index] for t in self.model_tensors])
        extra_tensors = tuple(t[index] for t in self.extra_tensors)
        return model_tensors, extra_tensors

    def __len__(self):
        return len(self.model_tensors[0])


class BasicDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_paths: list[str],
        batch_size: int = 64,
        num_workers: int = 4,
        sv_rate: float = 0.0,
        sv_seed: int = 0,
        dm_ic_enc_seq_len: int = 0,
    ):
        super().__init__()
        self.save_hyperparameters()

    def setup(self, stage=None):
        hps = self.hparams
        data_dicts = []
        for data_path in hps.data_paths:
            # Load data arrays from the file
            with h5py.File(data_path, "r") as h5file:
                data_dict = {k: v[()] for k, v in h5file.items()}
            # Add separate keys for encod and recon data
            data_dict.update(
                {
                    "train_encod_data": data_dict["train_data"],
                    "train_recon_data": data_dict["train_data"],
                    "valid_encod_data": data_dict["valid_data"],
                    "valid_recon_data": data_dict["valid_data"],
                }
            )
            data_dicts.append(data_dict)
        # Attach data to the datamodule
        attach_tensors(self, data_dicts)

    def train_dataloader(self, shuffle=True):
        # PTL provides all batches at once, so divide amongst dataloaders
        batch_size = int(self.hparams.batch_size / len(self.train_ds))
        dataloaders = {
            i: DataLoader(
                ds,
                batch_size=batch_size,
                num_workers=self.hparams.num_workers,
                shuffle=shuffle,
                drop_last=True,
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
