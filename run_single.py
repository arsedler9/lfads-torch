import hydra
from omegaconf import DictConfig, OmegaConf


@hydra.main(config_path="config_run/", config_name="single_run.yaml")
def main(config: DictConfig):

    # Imports should be nested inside @hydra.main to optimize tab completion
    # Read more here: https://github.com/facebookresearch/hydra/issues/934
    print(OmegaConf.to_yaml(config, resolve=True))
    from lfads_torch.run_model import run_model

    # Clear the GlobalHydra instance so we can compose again in `train`
    hydra.core.global_hydra.GlobalHydra.instance().clear()
    # Run the training function
    return run_model({"config_train": config.config_train})


if __name__ == "__main__":
    main()
