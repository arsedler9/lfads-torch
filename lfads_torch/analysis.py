import logging

import h5py
import numpy as np
import torch
from tqdm import tqdm

from .utils import batch_fwd

logger = logging.getLogger(__name__)


def run_posterior_sampling(model, trainer, filename, best_ckpt=False, num_samples=50):
    """Runs the model repeatedly to generate outputs for different samples
    of the posteriors. Averages these outputs and saves them to an output file.

    Parameters
    ----------
    model : lfads_torch.models.base_model.LFADS
        A trained LFADS model.
    trainer : pytorch_lightning.Trainer
        The `Trainer` used to train the `model`.
    filename : str
        The filename to use for saving output
    best_ckpt : bool, optional
        Whether to use the checkpoint with the lowest validation reconstruction
        cost, by default False uses the most recent checkpoint
    num_samples : int, optional
        The number of forward passes to average, by default 50
    """
    if best_ckpt:
        # Restore the best checkpoint if necessary
        best_model_path = trainer.checkpoint_callback.best_model_path
        model.load_from_checkpoint(best_model_path)
    # Batch the data
    train_dl = trainer.datamodule.train_dataloader(shuffle=False)
    valid_dl = trainer.datamodule.val_dataloader()

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
