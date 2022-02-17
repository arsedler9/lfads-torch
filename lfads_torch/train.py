import logging
from os import path
from typing import List, Optional

import hydra
import pytorch_lightning as pl
import torch
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.loggers import LightningLoggerBase

from .utils import flatten

log = logging.getLogger(__name__)


def train(overrides: dict, checkpoint_dir: str = None) -> Optional[float]:
    """Contains training pipeline.
    Instantiates all PyTorch Lightning objects from config.
    Args:
        config (DictConfig): Configuration composed by Hydra.
    Returns:
        Optional[float]: Metric score for hyperparameter optimization.
    """

    # Get the name of the train config
    config_train = overrides.pop("config_train")

    # Format the overrides so they can be used by hydra
    overrides = [f"{k}={v}" for k, v in flatten(overrides).items()]

    # Compose the train config
    config_path = path.join(path.dirname(path.dirname(__file__)), "configs/")
    with hydra.initialize(config_path=path.relpath(config_path), job_name="train"):
        config = hydra.compose(config_name=config_train, overrides=overrides)

    # Set seed for random number generators in pytorch, numpy and python.random
    if config.get("seed") is not None:
        pl.seed_everything(config.seed, workers=True)

    # Init lightning datamodule
    log.info(f"Instantiating datamodule <{config.datamodule._target_}>")
    datamodule: pl.LightningDataModule = hydra.utils.instantiate(config.datamodule)

    # Init lightning model
    log.info(f"Instantiating model <{config.model._target_}>")
    model: pl.LightningModule = hydra.utils.instantiate(config.model)

    # Init lightning callbacks
    callbacks: List[pl.Callback] = []
    if "callbacks" in config:
        for _, cb_conf in config.callbacks.items():
            if "_target_" in cb_conf:
                log.info(f"Instantiating callback <{cb_conf._target_}>")
                callbacks.append(hydra.utils.instantiate(cb_conf, _convert_="all"))

    # Init lightning loggers
    logger: List[LightningLoggerBase] = []
    if "logger" in config:
        for _, lg_conf in config.logger.items():
            if "_target_" in lg_conf:
                log.info(f"Instantiating logger <{lg_conf._target_}>")
                logger.append(hydra.utils.instantiate(lg_conf))

    # Init lightning trainer
    log.info(f"Instantiating trainer <{config.trainer._target_}>")
    resume_from_checkpoint = (
        path.join(checkpoint_dir, "checkpoint") if checkpoint_dir is not None else None
    )
    trainer: pl.Trainer = hydra.utils.instantiate(
        config.trainer,
        gpus=int(torch.cuda.is_available()),
        callbacks=callbacks,
        logger=logger,
        resume_from_checkpoint=resume_from_checkpoint,
        _convert_="partial",
    )

    # Train the model
    log.info("Starting training.")
    trainer.fit(model=model, datamodule=datamodule)
