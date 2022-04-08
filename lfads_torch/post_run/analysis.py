import logging

import h5py
import numpy as np
import torch
from tqdm import tqdm

from ..utils.misc import batch_fwd

logger = logging.getLogger(__name__)


def run_posterior_sampling(model, datamodule, filename, num_samples=50):
    """Runs the model repeatedly to generate outputs for different samples
    of the posteriors. Averages these outputs and saves them to an output file.

    Parameters
    ----------
    model : lfads_torch.models.base_model.LFADS
        A trained LFADS model.
    datamodule : pytorch_lightning.LightningDataModule
        The `LightningDataModule` to pass through the `model`.
    filename : str
        The filename to use for saving output
    num_samples : int, optional
        The number of forward passes to average, by default 50
    """
    # Batch the data
    datamodule.setup()
    train_dl = datamodule.train_dataloader(shuffle=False)
    valid_dl = datamodule.val_dataloader()

    def transpose_list(output):
        # Transpose a list of lists
        return list(map(list, zip(*output)))

    def run_ps_epoch(dataloader):
        # Compute all model outputs for the dataloader
        output = [
            batch_fwd(model, batch, sample_posteriors=True) for batch in dataloader
        ]
        # Concatenate outputs along the batch dimension
        return [torch.cat(o).detach().cpu().numpy() for o in transpose_list(output)]

    # Repeatedly get model output for the complete dataset
    logger.info("Running posterior sampling on train data.")
    train_ps = [run_ps_epoch(train_dl) for _ in tqdm(range(num_samples))]
    logger.info("Running posterior sampling on valid data.")
    valid_ps = [run_ps_epoch(valid_dl) for _ in tqdm(range(num_samples))]
    # Average across the samples
    train_pm = [np.mean(np.stack(o), axis=0) for o in transpose_list(train_ps)]
    valid_pm = [np.mean(np.stack(o), axis=0) for o in transpose_list(valid_ps)]
    # Save the averages to the output file
    with h5py.File(filename, mode="w") as h5file:
        for prefix, pm in zip(["train_", "valid_"], [train_pm, valid_pm]):
            for name, data in zip(
                [
                    "output_params",
                    "factors",
                    "ic_mean",
                    "ic_std",
                    "co_means",
                    "co_stds",
                    "gen_states",
                    "gen_init",
                    "gen_inputs",
                    "con_states",
                ],
                pm,
            ):
                h5file.create_dataset(prefix + name, data=data)
    # Log message about sucessful completion
    logger.info(f"Posterior averages successfully saved to `{filename}`")
