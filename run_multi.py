import logging

import hydra
import ray
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

logger = logging.getLogger(__name__)


@hydra.main(config_path="config_run/", config_name="multi_run.yaml")
def main(config: DictConfig):
    print(OmegaConf.to_yaml(config))
    from lfads_torch.train import train

    # Instantiate the scheduler
    scheduler = instantiate(config.scheduler, _convert_="all")
    # Instantiate the progress reporter
    progress_reporter = instantiate(config.progress_reporter, _convert_="all")
    # Instantiate the config search space
    search_space = instantiate(config.search_space, _convert_="all")
    # Specify the train config to use
    search_space["config_train"] = config.config_train
    # Clear the GlobalHydra instance so we can compose again in `train`
    hydra.core.global_hydra.GlobalHydra.instance().clear()
    # Run the search with `ray.tune`
    # ray.init(local_mode=True)
    analysis = ray.tune.run(
        train,
        config=search_space,
        scheduler=scheduler,
        progress_reporter=progress_reporter,
        **config["ray_tune_run"],
    )
    print(f"Best hyperparameters: {analysis.best_config}")


if __name__ == "__main__":
    main()
