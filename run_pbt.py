import logging
import shutil
from os import path

import hydra
import ray
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

logger = logging.getLogger(__name__)


@hydra.main(config_path="config_run/", config_name="pbt_run.yaml")
def main(config: DictConfig):
    print(OmegaConf.to_yaml(config))
    from lfads_torch.run_model import run_model

    # Instantiate the scheduler
    scheduler = instantiate(config.scheduler, _convert_="all")
    # Instantiate the config search space
    search_space = instantiate(config.search_space, _convert_="all")
    # Specify the train config to use
    search_space["config_train"] = config.config_train
    # Clear the GlobalHydra instance so we can compose again in `train`
    hydra.core.global_hydra.GlobalHydra.instance().clear()
    # Run the search with `ray.tune`
    ray_tune_run_params = instantiate(config["ray_tune_run"])
    # ray.init(local_mode=True)
    analysis = ray.tune.run(
        ray.tune.with_parameters(run_model, do_posterior_sample=False),
        config=search_space,
        scheduler=scheduler,
        **ray_tune_run_params,
    )
    print(f"Best hyperparameters: {analysis.best_config}")
    # Load the best model and run posterior sampling (skip training)
    best_model_dir = path.join(path.dirname(analysis.best_logdir), "best_model")
    shutil.copytree(analysis.best_logdir, best_model_dir)
    overrides = {"config_train": config.config_train}
    run_model(overrides, checkpoint_dir=best_model_dir, do_train=False)


if __name__ == "__main__":
    main()
