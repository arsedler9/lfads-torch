import io
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pytorch_lightning as pl
import torch
from nlb_tools.evaluation import (
    bits_per_spike,
    eval_psth,
    speed_tp_correlation,
    velocity_decoding,
)
from PIL import Image
from scipy.linalg import LinAlgWarning
from sklearn.decomposition import PCA

from .metrics import ExpSmoothedMetric
from .utils import send_batch_to_device

plt.switch_backend("Agg")


def has_image_loggers(loggers):
    """Checks whether any image loggers are available.

    Parameters
    ----------
    loggers : obj or list[obj]
        An object or list of loggers to search.
    """
    logger_list = loggers if isinstance(loggers, list) else [loggers]
    for logger in logger_list:
        if isinstance(logger, pl.loggers.TensorBoardLogger):
            return True
        elif isinstance(logger, pl.loggers.WandbLogger):
            return True
    return False


def log_figure(loggers, name, fig, step):
    """Logs a figure image to all available image loggers.

    Parameters
    ----------
    loggers : obj or list[obj]
        An object or list of loggers
    name : str
        The name to use for the logged figure
    fig : matplotlib.figure.Figure
        The figure to log
    step : int
        The step to associate with the logged figure
    """
    # Save figure image to in-memory buffer
    img_buf = io.BytesIO()
    fig.savefig(img_buf, format="png")
    image = Image.open(img_buf)
    # Distribute image to all image loggers
    logger_list = loggers if isinstance(loggers, list) else [loggers]
    for logger in logger_list:
        if isinstance(logger, pl.loggers.TensorBoardLogger):
            logger.experiment.add_figure(name, fig, step)
        elif isinstance(logger, pl.loggers.WandbLogger):
            logger.log_image(name, [image], step)
    img_buf.close()


class RasterPlot(pl.Callback):
    """Plots validation spiking data side-by-side with
    inferred inputs and rates and logs to tensorboard.
    """

    def __init__(self, split="valid", n_samples=3, log_every_n_epochs=100):
        """Initializes the callback.
        Parameters
        ----------
        n_samples : int, optional
            The number of samples to plot, by default 3
        log_every_n_epochs : int, optional
            The frequency with which to plot and log, by default 100
        """
        assert split in ["train", "valid"]
        self.split = split
        self.n_samples = n_samples
        self.log_every_n_epochs = log_every_n_epochs

    def on_validation_epoch_end(self, trainer, pl_module):
        """Logs plots at the end of the validation epoch.
        Parameters
        ----------
        trainer : pytorch_lightning.Trainer
            The trainer currently handling the model.
        pl_module : pytorch_lightning.LightningModule
            The model currently being trained.
        """
        if (trainer.current_epoch % self.log_every_n_epochs) != 0:
            return
        # Check for any image loggers
        if not has_image_loggers(trainer.loggers):
            return
        # Get data samples from the dataloaders
        if self.split == "valid":
            dataloader = trainer.datamodule.val_dataloader()
        else:
            dataloader = trainer.datamodule.train_dataloader(shuffle=False)
        batch = next(iter(dataloader))
        # Determine which sessions are in the batch
        sessions = sorted(batch.keys())
        # Move data to the right device
        batch = send_batch_to_device(batch, pl_module.device)
        # Compute model output
        output = pl_module.predict_step(
            batch=batch,
            batch_ix=None,
            sample_posteriors=False,
        )
        # Discard the extra data - only the SessionBatches are relevant here
        batch = {s: b[0] for s, b in batch.items()}
        # Log a few example outputs for each session
        for s in sessions:
            # Convert everything to numpy
            encod_data = batch[s].encod_data.detach().cpu().numpy()
            recon_data = batch[s].recon_data.detach().cpu().numpy()
            truth = batch[s].truth.detach().cpu().numpy()
            means = output[s].output_params.detach().cpu().numpy()
            inputs = output[s].gen_inputs.detach().cpu().numpy()
            # Compute data sizes
            _, steps_encod, neur_encod = encod_data.shape
            _, steps_recon, neur_recon = recon_data.shape
            # Decide on how to plot panels
            if np.all(np.isnan(truth)):
                plot_arrays = [recon_data, means, inputs]
                height_ratios = [3, 3, 1]
            else:
                plot_arrays = [recon_data, truth, means, inputs]
                height_ratios = [3, 3, 3, 1]
            # Create subplots
            fig, axes = plt.subplots(
                len(plot_arrays),
                self.n_samples,
                sharex=True,
                sharey="row",
                figsize=(3 * self.n_samples, 10),
                gridspec_kw={"height_ratios": height_ratios},
            )
            for i, ax_col in enumerate(axes.T):
                for j, (ax, array) in enumerate(zip(ax_col, plot_arrays)):
                    if j < len(plot_arrays) - 1:
                        ax.imshow(array[i].T, interpolation="none", aspect="auto")
                        ax.vlines(steps_encod, 0, neur_recon, color="orange")
                        ax.hlines(neur_encod, 0, steps_recon, color="orange")
                        ax.set_xlim(0, steps_recon)
                        ax.set_ylim(0, neur_recon)
                    else:
                        ax.plot(array[i])
            plt.tight_layout()
            # Log the figure
            log_figure(
                trainer.loggers,
                f"{self.split}/raster_plot/sess{s}",
                fig,
                trainer.global_step,
            )


class TrajectoryPlot(pl.Callback):
    """Plots the top-3 PC's of the latent trajectory for
    all samples in the validation set and logs to tensorboard.
    """

    def __init__(self, log_every_n_epochs=100):
        """Initializes the callback.

        Parameters
        ----------
        log_every_n_epochs : int, optional
            The frequency with which to plot and log, by default 100
        """
        self.log_every_n_epochs = log_every_n_epochs

    def on_validation_epoch_end(self, trainer, pl_module):
        """Logs plots at the end of the validation epoch.

        Parameters
        ----------
        trainer : pytorch_lightning.Trainer
            The trainer currently handling the model.
        pl_module : pytorch_lightning.LightningModule
            The model currently being trained.
        """
        # Skip evaluation for most epochs to save time
        if (trainer.current_epoch % self.log_every_n_epochs) != 0:
            return
        # Check for any image loggers
        if not has_image_loggers(trainer.loggers):
            return
        # Get only the validation dataloaders
        pred_dls = trainer.datamodule.predict_dataloader()
        dataloaders = {s: dls["valid"] for s, dls in pred_dls.items()}
        # Compute outputs and plot for one session at a time
        for s, dataloader in dataloaders.items():
            latents = []
            for batch in dataloader:
                # Move data to the right device
                batch = send_batch_to_device({s: batch}, pl_module.device)
                # Perform the forward pass through the model
                output = pl_module.predict_step(batch, None, sample_posteriors=False)[s]
                latents.append(output.factors)
            latents = torch.cat(latents).detach().cpu().numpy()
            # Reduce dimensionality if necessary
            n_samp, n_step, n_lats = latents.shape
            if n_lats > 3:
                latents_flat = latents.reshape(-1, n_lats)
                pca = PCA(n_components=3)
                latents = pca.fit_transform(latents_flat)
                latents = latents.reshape(n_samp, n_step, 3)
                explained_variance = np.sum(pca.explained_variance_ratio_)
            else:
                explained_variance = 1.0
            # Create figure and plot trajectories
            fig = plt.figure(figsize=(10, 10))
            ax = fig.add_subplot(111, projection="3d")
            for traj in latents:
                ax.plot(*traj.T, alpha=0.2, linewidth=0.5)
            ax.scatter(*latents[:, 0, :].T, alpha=0.1, s=10, c="g")
            ax.scatter(*latents[:, -1, :].T, alpha=0.1, s=10, c="r")
            ax.set_title(f"explained variance: {explained_variance:.2f}")
            plt.tight_layout()
            # Log the figure
            log_figure(
                trainer.loggers,
                f"trajectory_plot/sess{s}",
                fig,
                trainer.global_step,
            )


class NLBEvaluation(pl.Callback):
    """Computes and logs all evaluation metrics for the Neural Latents
    Benchmark to tensorboard. These include `co_bps`, `fp_bps`,
    `behavior_r2`, `psth_r2`, and `tp_corr`.
    """

    def __init__(self, log_every_n_epochs=20, decoding_cv_sweep=False):
        """Initializes the callback.

        Parameters
        ----------
        log_every_n_epochs : int, optional
            The frequency with which to plot and log, by default 100
        decoding_cv_sweep : bool, optional
            Whether to run a cross-validated hyperparameter sweep to
            find optimal regularization values, by default False
        """
        self.log_every_n_epochs = log_every_n_epochs
        self.decoding_cv_sweep = decoding_cv_sweep
        self.smth_metrics = {}

    def on_validation_epoch_end(self, trainer, pl_module):
        """Logs plots at the end of the validation epoch.

        Parameters
        ----------
        trainer : pytorch_lightning.Trainer
            The trainer currently handling the model.
        pl_module : pytorch_lightning.LightningModule
            The model currently being trained.
        """
        # Skip evaluation for most epochs to save time
        if (trainer.current_epoch % self.log_every_n_epochs) != 0:
            return
        # Get the dataloaders
        pred_dls = trainer.datamodule.predict_dataloader()
        s = 0
        val_dataloader = pred_dls[s]["valid"]
        train_dataloader = pred_dls[s]["train"]
        # Create object to store evaluation metrics
        metrics = {}
        # Get entire validation dataset from datamodule
        (input_data, recon_data, *_), (behavior,) = trainer.datamodule.valid_data[s]
        recon_data = recon_data.detach().cpu().numpy()
        behavior = behavior.detach().cpu().numpy()
        # Pass the data through the model
        # TODO: Replace this with Trainer.predict? Hesitation is that switching to
        # Trainer.predict for posterior sampling is inefficient because we can't
        # tell it how many forward passes to use.
        rates = []
        for batch in val_dataloader:
            batch = send_batch_to_device({s: batch}, pl_module.device)
            output = pl_module.predict_step(batch, None, sample_posteriors=False)[s]
            rates.append(output.output_params)
        rates = torch.cat(rates).detach().cpu().numpy()
        # Compute co-smoothing bits per spike
        _, n_obs, n_heldin = input_data.shape
        heldout = recon_data[:, :n_obs, n_heldin:]
        rates_heldout = rates[:, :n_obs, n_heldin:]
        co_bps = bits_per_spike(rates_heldout, heldout)
        metrics["nlb/co_bps"] = max(co_bps, -1.0)
        # Compute forward prediction bits per spike
        forward = recon_data[:, n_obs:]
        rates_forward = rates[:, n_obs:]
        fp_bps = bits_per_spike(rates_forward, forward)
        metrics["nlb/fp_bps"] = max(fp_bps, -1.0)
        # Get relevant training dataset from datamodule
        _, (train_behavior,) = trainer.datamodule.train_data[s]
        train_behavior = train_behavior.detach().cpu().numpy()
        # Get model predictions for the training dataset
        train_rates = []
        for batch in train_dataloader:
            batch = send_batch_to_device({s: batch}, pl_module.device)
            output = pl_module.predict_step(batch, None, sample_posteriors=False)[s]
            train_rates.append(output.output_params)
        train_rates = torch.cat(train_rates).detach().cpu().numpy()
        # Get firing rates for observed time points
        rates_obs = rates[:, :n_obs]
        train_rates_obs = train_rates[:, :n_obs]
        # Compute behavioral decoding performance
        if "dmfc_rsg" in trainer.datamodule.hparams.dataset_name:
            tp_corr = speed_tp_correlation(heldout, rates_obs, behavior)
            metrics["nlb/tp_corr"] = tp_corr
        else:
            with warnings.catch_warnings():
                # Ignore LinAlgWarning from early in training
                warnings.filterwarnings("ignore", category=LinAlgWarning)
                behavior_r2 = velocity_decoding(
                    train_rates_obs,
                    train_behavior,
                    trainer.datamodule.train_decode_mask,
                    rates_obs,
                    behavior,
                    trainer.datamodule.valid_decode_mask,
                    self.decoding_cv_sweep,
                )
            metrics["nlb/behavior_r2"] = max(behavior_r2, -1.0)
        # Compute PSTH reconstruction performance
        if hasattr(trainer.datamodule, "psth"):
            psth = trainer.datamodule.psth
            cond_idxs = trainer.datamodule.valid_cond_idxs
            jitter = trainer.datamodule.valid_jitter
            psth_r2 = eval_psth(psth, rates_obs, cond_idxs, jitter)
            metrics["nlb/psth_r2"] = max(psth_r2, -1.0)
        # Compute smoothed metrics
        for k, v in metrics.items():
            if k not in self.smth_metrics:
                self.smth_metrics[k] = ExpSmoothedMetric(coef=0.7)
            self.smth_metrics[k].update(v, 1)
        # Log actual and smoothed metrics
        pl_module.log_dict(
            {
                **metrics,
                **{k + "_smth": m.compute() for k, m in self.smth_metrics.items()},
            }
        )
        # Reset the smoothed metrics (per-step aggregation not necessary)
        [m.reset() for m in self.smth_metrics.values()]
