import logging
import os
import warnings
from glob import glob
from pathlib import Path

import hydra
import pytorch_lightning as pl
import torch
from hydra.utils import call, instantiate

from .utils import flatten

log = logging.getLogger(__name__)


def run_model(
    overrides: dict = {},
    checkpoint_dir: str = None,
    config_path: str = "../configs/single.yaml",
    do_train: bool = True,
    do_posterior_sample: bool = True,
):
    """Adds overrides to the default config, instantiates all PyTorch Lightning
    objects from config, and runs the training pipeline.
    """

    # Compose the train config with properly formatted overrides
    config_path = Path(config_path)
    overrides = [f"{k}={v}" for k, v in flatten(overrides).items()]
    with hydra.initialize(config_path=config_path.parent, job_name="run_model"):
        config = hydra.compose(config_name=config_path.name, overrides=overrides)

    # Avoid flooding the console with output during multi-model runs
    if config.ignore_warnings:
        logging.getLogger("pytorch_lightning").setLevel(logging.WARNING)
        warnings.filterwarnings("ignore")

    # Set seed for random number generators in pytorch, numpy and python.random
    if config.get("seed") is not None:
        pl.seed_everything(config.seed, workers=True)

    # Instantiate `LightningDataModule` and `LightningModule`
    datamodule = instantiate(config.datamodule)
    model = instantiate(config.model)

    # If `checkpoint_dir` is passed, find the most recent checkpoint in the directory
    if checkpoint_dir:
        ckpt_pattern = os.path.join(checkpoint_dir, "*.ckpt")
        ckpt_path = max(glob(ckpt_pattern), key=os.path.getctime)
        # Overwrite any `optimizer_states` so optimizers are not restored by PTL
        ckpt = torch.load(ckpt_path)
        ckpt["optimizer_states"] = []
        torch.save(ckpt, ckpt_path)

    if do_train:
        # Instantiate the pytorch_lightning `Trainer` and its callbacks and loggers
        trainer = instantiate(
            config.trainer,
            callbacks=[instantiate(c) for c in config.callbacks.values()],
            logger=[instantiate(lg) for lg in config.logger.values()],
            gpus=int(torch.cuda.is_available()),
        )
        # Train the model
        trainer.fit(
            model=model,
            datamodule=datamodule,
            ckpt_path=ckpt_path if checkpoint_dir else None,
        )
        # Restore the best checkpoint if necessary - otherwise, use last checkpoint
        if config.posterior_sampling.use_best_ckpt:
            ckpt_path = trainer.checkpoint_callback.best_model_path
            model.load_state_dict(torch.load(ckpt_path)["state_dict"])
    else:
        if checkpoint_dir:
            # If not training, restore model from the checkpoint
            model.load_state_dict(torch.load(ckpt_path)["state_dict"])

    # Run the posterior sampling function
    if do_posterior_sample:
        # TODO: Make more efficient by single transfer to GPU
        call(config.posterior_sampling.fn, model=model, datamodule=datamodule)
