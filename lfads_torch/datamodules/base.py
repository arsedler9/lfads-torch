import h5py
import pytorch_lightning as pl
import torch
from pytorch_lightning.trainer.supporters import CombinedLoader
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

from ..tuples import SessionBatch

MANDATORY_KEYS = {
    "train": ["encod_data", "recon_data"],
    "valid": ["encod_data", "recon_data"],
    "test": ["encod_data", "recon_data"],
}


def to_tensor(array):
    return torch.tensor(array, dtype=torch.float)


def attach_tensors(datamodule, data_dicts: list[dict], extra_keys: list[str] = []):
    hps = datamodule.hparams
    sv_gen = torch.Generator().manual_seed(hps.sv_seed)
    all_train_data, all_valid_data, all_test_data = [], [], []
    for data_dict in data_dicts:

        def create_session_batch(prefix):
            # Ensure that the data dict has all of the required keys
            assert all(f"{prefix}_{key}" in data_dict for key in MANDATORY_KEYS[prefix])
            encod_data = to_tensor(data_dict[f"{prefix}_encod_data"])
            recon_data = to_tensor(data_dict[f"{prefix}_recon_data"])
            # Create sample validation mask
            bern_p = 1 - hps.sv_rate if prefix != "test" else 1.0
            sv_mask = torch.bernoulli(encod_data, p=bern_p, generator=sv_gen)
            # Load or simulate external inputs
            if f"{prefix}_ext_input" in data_dict:
                ext_input = to_tensor(data_dict[f"{prefix}_ext_input"])
            else:
                ext_input = torch.zeros_like(encod_data[..., :0])
            # Load or simulate ground truth
            if f"{prefix}_truth" in data_dict:
                cf = data_dict["conversion_factor"]
                truth = to_tensor(data_dict[f"{prefix}_truth"]) / cf
            else:
                truth = torch.full_like(recon_data, float("nan"))
            # Remove unnecessary data during IC encoder segment
            sv_mask = sv_mask[:, hps.dm_ic_enc_seq_len :]
            ext_input = ext_input[:, hps.dm_ic_enc_seq_len :]
            truth = truth[:, hps.dm_ic_enc_seq_len :, :]
            # Extract data for any extra keys
            other = [to_tensor(data_dict[f"{prefix}_{k}"]) for k in extra_keys]
            return (
                SessionBatch(
                    encod_data=encod_data,
                    recon_data=recon_data,
                    ext_input=ext_input,
                    truth=truth,
                    sv_mask=sv_mask,
                ),
                tuple(other),
            )

        # Store the data for each session
        all_train_data.append(create_session_batch("train"))
        all_valid_data.append(create_session_batch("valid"))
        if "test_encod_data" in data_dict:
            all_test_data.append(create_session_batch("test"))
    # Store the datasets on the datamodule
    datamodule.train_data = all_train_data
    datamodule.train_ds = [SessionDataset(*train_data) for train_data in all_train_data]
    datamodule.valid_data = all_valid_data
    datamodule.valid_ds = [SessionDataset(*valid_data) for valid_data in all_valid_data]
    if len(all_test_data) == len(all_train_data):
        datamodule.test_data = all_test_data
        datamodule.test_ds = [SessionDataset(*test_data) for test_data in all_test_data]


def reshuffle_train_valid(train_tensors, valid_tensors, seed):
    n_train, n_valid = len(train_tensors[0]), len(valid_tensors[0])
    tensors = [torch.cat([t, v]) for t, v in zip(train_tensors, valid_tensors)]
    gen = torch.Generator().manual_seed(seed)
    inds = torch.randperm(n_train + n_valid, generator=gen)
    train_inds, valid_inds = torch.split(inds, [n_train, n_valid])
    train_tensors = [t[train_inds] for t in tensors]
    valid_tensors = [t[valid_inds] for t in tensors]
    return train_tensors, valid_tensors, train_inds, valid_inds


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
        reshuffle_tv_seed: int = None,
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
            if "train_data" in data_dict:
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
        if hps.reshuffle_tv_seed is not None:
            # Reshuffle the training / validation split
            self.train_data, self.valid_data, _, _ = reshuffle_train_valid(
                train_tensors=self.train_data,
                valid_tensors=self.valid_data,
                seed=hps.reshuffle_tv_seed,
            )

    def train_dataloader(self, shuffle=True):
        # PTL provides all batches at once, so divide amongst dataloaders
        batch_size = int(self.hparams.batch_size / len(self.train_ds))
        dataloaders = {
            i: DataLoader(
                ds,
                batch_size=batch_size,
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
                    shuffle=False,
                ),
                "valid": DataLoader(
                    self.valid_ds[s],
                    batch_size=self.hparams.batch_size,
                    shuffle=False,
                ),
            }
            for s in range(len(self.train_ds))
        }
        # Add the test dataset if it is available
        if hasattr(self, "test_ds"):
            for s in range(len(self.train_ds)):
                dataloaders[s]["test"] = DataLoader(
                    self.test_ds[s],
                    batch_size=self.hparams.batch_size,
                    shuffle=False,
                )
        return dataloaders
