import os
import shutil
import sys
from pathlib import Path

import yaml
from hydra.utils import instantiate
from ray import tune
from ray.tune import CLIReporter
from ray.tune.search.basic_variant import BasicVariantGenerator

from lfads_torch.extensions.tune import BinaryTournamentPBT, ImprovementRatioStopper
from lfads_torch.run_model import run_model
from lfads_torch.utils import cleanup_best_model, read_pbt_fitlog


# Function to keep dropout and CD rates in-bounds
def clip_config_rates(config):
    return {k: min(v, 0.99) if "_rate" in k else v for k, v in config.items()}


# Get path arguments provided by NeuroCAAS
_, datapath, configpath, resultpath = sys.argv
print(f"--Data: {datapath} Config: {configpath} Results: {resultpath}--")
# Load the YAML file to get PBT parameters
with open(configpath) as yfile:
    cfg = yaml.safe_load(yfile)["pbt"]
# Instantiate hyperparameters
hyperparam_space = instantiate(cfg["hps"])
init_space = {name: tune.sample_from(hp.init) for name, hp in hyperparam_space.items()}
# Run the hyperparameter search
analysis = tune.run(
    tune.with_parameters(
        run_model,
        config_path=configpath,
        do_posterior_sample=False,
    ),
    metric=cfg["metric"],
    mode="min",
    name=Path(resultpath).name,
    stop=ImprovementRatioStopper(
        num_trials=cfg["num_samples"],
        perturbation_interval=cfg["perturbation_interval"],
        burn_in_period=cfg["burn_in_period"],
        metric=cfg["metric"],
        patience=cfg["patience"],
        min_improvement_ratio=cfg["min_improvement_ratio"],
    ),
    config={"datamodule.data_paths.0": datapath, **init_space},
    resources_per_trial=cfg["resources_per_trial"],
    num_samples=cfg["num_samples"],
    local_dir=Path(resultpath).parent,
    search_alg=BasicVariantGenerator(random_state=0),
    scheduler=BinaryTournamentPBT(
        perturbation_interval=cfg["perturbation_interval"],
        burn_in_period=cfg["burn_in_period"],
        hyperparam_mutations=hyperparam_space,
    ),
    keep_checkpoints_num=1,
    verbose=1,
    progress_reporter=CLIReporter(
        metric_columns=[cfg["metric"], "cur_epoch"],
        sort_by_metric=True,
    ),
    trial_dirname_creator=lambda trial: str(trial),
)
# Copy the best model to a new folder so it is easy to identify
best_model_dir = Path(resultpath) / "best_model"
shutil.copytree(analysis.best_logdir, best_model_dir)
# Switch working directory to this folder (usually handled by tune)
os.chdir(best_model_dir)
# Load the best model and run posterior sampling (skip training)
best_ckpt_dir = best_model_dir / Path(analysis.best_checkpoint._local_path).name
run_model(
    checkpoint_dir=best_ckpt_dir,
    config_path=configpath,
    do_train=False,
    overrides={"datamodule.data_paths.0": datapath},
)
# Assemble training / hyperparameter log
fit_df = read_pbt_fitlog(resultpath)
fit_df.to_csv(best_model_dir / "fitlog.csv")
# Remove extra files
cleanup_best_model(str(best_model_dir))
