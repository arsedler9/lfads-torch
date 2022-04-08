import logging
import os
import warnings
from glob import glob
from typing import List

import pytorch_lightning as pl
import torch
from hydra import compose, initialize
from hydra.utils import call, instantiate
from pytorch_lightning.loggers import LightningLoggerBase

from .utils import flatten

log = logging.getLogger(__name__)


def run_model(
    overrides: dict = {},
    checkpoint_dir: str = None,
    config_train: str = "single_train.yaml",
    do_train: bool = True,
    do_posterior_sample: bool = True,
):
    """Adds overrides to the default config, instantiates all PyTorch Lightning
    objects from config, and runs the training pipeline.
    """

    # Format the overrides so they can be used by hydra
    overrides = [f"{k}={v}" for k, v in flatten(overrides).items()]

    # Compose the train config
    with initialize(config_path="../configs/", job_name="run_model"):
        config = compose(config_name=config_train, overrides=overrides)

    # Avoid flooding the console with output during multi-model runs
    if config.ignore_warnings:
        logging.getLogger("pytorch_lightning").setLevel(logging.WARNING)
        warnings.filterwarnings("ignore")

    # Set seed for random number generators in pytorch, numpy and python.random
    if config.get("seed") is not None:
        pl.seed_everything(config.seed, workers=True)

    # Init lightning datamodule
    log.info(f"Instantiating datamodule <{config.datamodule._target_}>")
    datamodule: pl.LightningDataModule = instantiate(config.datamodule)

    # Init lightning model
    log.info(f"Instantiating model <{config.model._target_}>")
    model: pl.LightningModule = instantiate(config.model)

    # Set model checkpoint path if necessary
    if checkpoint_dir:
        ckpt_pattern = os.path.join(checkpoint_dir, "*.ckpt")
        ckpt_path = max(glob(ckpt_pattern), key=os.path.getctime)
        state_dict = torch.load(ckpt_path)["state_dict"]
        model.load_state_dict(state_dict)

    if do_train:
        # Init lightning callbacks
        callbacks: List[pl.Callback] = []
        if "callbacks" in config:
            for _, cb_conf in config.callbacks.items():
                if "_target_" in cb_conf:
                    log.info(f"Instantiating callback <{cb_conf._target_}>")
                    callbacks.append(instantiate(cb_conf, _convert_="all"))

        # Init lightning loggers
        logger: List[LightningLoggerBase] = []
        if "logger" in config:
            for _, lg_conf in config.logger.items():
                if "_target_" in lg_conf:
                    log.info(f"Instantiating logger <{lg_conf._target_}>")
                    logger.append(instantiate(lg_conf))

        # Init lightning trainer
        log.info(f"Instantiating trainer <{config.trainer._target_}>")
        trainer: pl.Trainer = instantiate(
            config.trainer,
            gpus=int(torch.cuda.is_available()),
            callbacks=callbacks,
            logger=logger,
            _convert_="partial",
        )
        # Train the model
        log.info("Starting training.")
        trainer.fit(model=model, datamodule=datamodule)
        # Restore the best checkpoint if necessary
        if config.posterior_sampling.best_ckpt:
            best_model_path = trainer.checkpoint_callback.best_model_path
            state_dict = torch.load(best_model_path)["state_dict"]
            model.load_state_dict(state_dict)

    # Run the posterior sampling function
    if do_posterior_sample:
        call(config.posterior_sampling.fn, model=model, datamodule=datamodule)
