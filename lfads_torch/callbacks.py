import io

import matplotlib.pyplot as plt
import numpy as np
import pytorch_lightning as pl

plt.switch_backend("Agg")


def fig_to_rgb_array(fig):
    """Converts a matplotlib figure into an array
    that can be logged to tensorboard.
    Parameters
    ----------
    fig : matplotlib.figure.Figure
        The figure to be converted.
    Returns
    -------
    np.array
        The figure as an HxWxC array of pixel values.
    """
    # Convert the figure to a numpy array
    with io.BytesIO() as buff:
        fig.savefig(buff, format="raw")
        buff.seek(0)
        fig_data = np.frombuffer(buff.getvalue(), dtype=np.uint8)
    w, h = fig.canvas.get_width_height()
    im = fig_data.reshape((int(h), int(w), -1))
    plt.close()
    return im


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
        # Get data samples
        dataloader = trainer.datamodule.val_dataloader()
        data, sv_mask, ext, truth = next(iter(dataloader))
        # Compute model output
        means, *_, inputs, _ = pl_module(
            data.to(pl_module.device),
            ext.to(pl_module.device),
            sample_posteriors=False,
        )
        # Convert everything to numpy
        data = data.detach().cpu().numpy()
        truth = truth.detach().cpu().numpy()
        means = means.detach().cpu().numpy()
        inputs = inputs.detach().cpu().numpy()
        if np.all(np.isnan(truth)):
            plot_arrays = [data, means, inputs]
            height_ratios = [3, 3, 1]
        else:
            plot_arrays = [data, truth, means, inputs]
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
            for ax, array in zip(ax_col, plot_arrays):
                ax.imshow(array[i].T, interpolation="none", aspect="auto")
        plt.tight_layout()
        # Log the plot to tensorboard
        im = fig_to_rgb_array(fig)
        # TODO: make this more robust
        trainer.logger.experiment[1].add_image(
            "raster_plot", im, trainer.global_step, dataformats="HWC"
        )
