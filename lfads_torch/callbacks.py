import matplotlib.pyplot as plt
import numpy as np
import pytorch_lightning as pl

plt.switch_backend("Agg")


def get_tensorboard_summary_writer(loggers):
    """Gets the TensorBoard SummaryWriter from a logger
    or logger collection to allow writing of images.

    Parameters
    ----------
    loggers : obj or list[obj]
        An object or list of loggers to search for the
        SummaryWriter.

    Returns
    -------
    torch.utils.tensorboard.writer.SummaryWriter
        The SummaryWriter object.
    """
    logger_list = loggers if isinstance(loggers, list) else [loggers]
    for logger in logger_list:
        if isinstance(logger, pl.loggers.tensorboard.TensorBoardLogger):
            return logger.experiment
    else:
        return None


class RasterPlot(pl.Callback):
    """Plots validation spiking data side-by-side with
    inferred inputs and rates and logs to tensorboard.
    """

    def __init__(self, n_samples=2, log_every_n_epochs=20):
        """Initializes the callback.
        Parameters
        ----------
        n_samples : int, optional
            The number of samples to plot, by default 2
        log_every_n_epochs : int, optional
            The frequency with which to plot and log, by default 20
        """
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
        # Check for the TensorBoard SummaryWriter
        writer = get_tensorboard_summary_writer(trainer.loggers)
        if writer is None:
            return
        # Get data samples
        dataloader = trainer.datamodule.val_dataloader()
        encod_data, recon_data, _, ext, truth, *_ = next(iter(dataloader))
        # Compute data sizes
        _, steps_encod, neur_encod = encod_data.shape
        _, steps_recon, neur_recon = recon_data.shape
        # Compute model output
        means, *_, inputs, _ = pl_module(
            encod_data.to(pl_module.device),
            ext.to(pl_module.device),
            sample_posteriors=False,
        )
        # Convert everything to numpy
        recon_data = recon_data.detach().cpu().numpy()
        truth = truth.detach().cpu().numpy()
        means = means.detach().cpu().numpy()
        inputs = inputs.detach().cpu().numpy()
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
        # Log the plot to tensorboard
        writer.add_figure("raster_plot", fig, trainer.global_step)
