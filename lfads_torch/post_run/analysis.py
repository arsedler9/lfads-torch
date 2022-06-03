import logging

import h5py
import torch
from tqdm import tqdm

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

    def run_ps_batch(batch):
        # Move the batch to the model device
        batch = [t.to(model.device) for t in batch]
        # Repeatedly compute the model outputs for this batch
        for i in range(num_samples):
            output = model.predict_step(batch, None, sample_posteriors=True)
            # Use running sum to save memory while averaging
            if i == 0:
                # Detach output from the graph to save memory on gradients
                sums = [o.detach() for o in output]
            else:
                sums = [s + o.detach() for s, o in zip(sums, output)]
        # Finish averaging by dividing by the total number of samples
        return [s / num_samples for s in sums]

    # Repeatedly get model output for the complete dataset
    logger.info("Running posterior sampling on train data.")
    train_ps = [run_ps_batch(batch) for batch in tqdm(train_dl)]
    logger.info("Running posterior sampling on valid data.")
    valid_ps = [run_ps_batch(batch) for batch in tqdm(valid_dl)]
    # Average across the samples
    train_pm = [torch.cat(o).cpu().numpy() for o in transpose_list(train_ps)]
    valid_pm = [torch.cat(o).cpu().numpy() for o in transpose_list(valid_ps)]
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
