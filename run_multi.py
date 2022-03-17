import logging
import os
import shutil

import hydra
import ray
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

logger = logging.getLogger(__name__)


@hydra.main(config_path="config_run/", config_name="multi_run.yaml")
def main(config: DictConfig):
    print(OmegaConf.to_yaml(config))
    from lfads_torch.run_model import run_model

    # Clear the GlobalHydra instance so we can compose again in `train`
    hydra.core.global_hydra.GlobalHydra.instance().clear()
    # Instantiate arguments to `ray.tune.run` can check here to debug
    ray_tune_run_kwargs = instantiate(config.ray_tune_run_kwargs, _convert_="all")
    # Enable local model for debugging
    if config.local_mode:
        ray.init(local_mode=True)
    # If overwriting, clear the working directory
    if config.overwrite:
        shutil.rmtree(os.getcwd() + "/", ignore_errors=True)
    # Run the experiment with `ray.tune`
    ray.tune.run(
        ray.tune.with_parameters(run_model, config_train=config.config_train),
        **ray_tune_run_kwargs,
    )


if __name__ == "__main__":
    main()
