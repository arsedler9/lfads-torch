import logging
from pathlib import Path

import h5py
import torch
from tqdm import tqdm

from ..tuples import SessionOutput
from ..utils import send_batch_to_device, transpose_lists

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
    # Convert filename to pathlib.Path for convenience
    filename = Path(filename)
    # Set up the dataloaders
    datamodule.setup()
    pred_dls = datamodule.predict_dataloader()

    # Function to run posterior sampling for a single session at a time
    def run_ps_batch(s, batch):
        # Move the batch to the model device
        batch = send_batch_to_device({s: batch}, model.device)
        # Repeatedly compute the model outputs for this batch
        for i in range(num_samples):
            # Perform the forward pass through the model
            output = model.predict_step(batch, None, sample_posteriors=True)[s]
            # Use running sum to save memory while averaging
            if i == 0:
                # Detach output from the graph to save memory on gradients
                sums = [o.detach() for o in output]
            else:
                sums = [s + o.detach() for s, o in zip(sums, output)]
        # Finish averaging by dividing by the total number of samples
        return [s / num_samples for s in sums]

    # Compute outputs for one session at a time
    for s, dataloaders in pred_dls.items():
        # Give each session a unique file path
        sess_fname = f"{filename.stem}_sess{s}{filename.suffix}"
        for split in ["train", "valid"]:
            # Compute average model outputs for each session and then recombine batches
            logger.info(f"Running posterior sampling on Session {s} {split} data.")
            post_means = [run_ps_batch(s, batch) for batch in tqdm(dataloaders[split])]
            post_means = SessionOutput(
                *[torch.cat(o).cpu().numpy() for o in transpose_lists(post_means)]
            )
            # Save the averages to the output file
            with h5py.File(sess_fname, mode="w") as h5file:
                for name in SessionOutput._fields:
                    h5file.create_dataset(
                        f"{split}_{name}", data=getattr(post_means, name)
                    )
        # Log message about sucessful completion
        logger.info(f"Session {s} posterior means successfully saved to `{sess_fname}`")
