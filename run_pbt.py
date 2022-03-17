import logging
import os
import shutil

import hydra
import ray
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

logger = logging.getLogger(__name__)


@hydra.main(config_path="config_run/", config_name="pbt_run.yaml")
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
    # Run the experiment with `ray.tune`, skip posterior sample to save space
    analysis = ray.tune.run(
        ray.tune.with_parameters(
            run_model,
            config_train=config.config_train,
            do_posterior_sample=False,
        ),
        **ray_tune_run_kwargs,
    )
    # Load the best model and run posterior sampling (skip training)
    best_model_dir = os.path.join(os.getcwd(), "best_model")
    shutil.copytree(analysis.best_logdir, best_model_dir)
    run_model(
        checkpoint_dir=best_model_dir,
        config_train=config.config_train,
        do_train=False,
    )


if __name__ == "__main__":
    main()
