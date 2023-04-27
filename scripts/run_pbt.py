import logging
import os
import shutil
import sys
from pathlib import Path

import yaml
from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import PopulationBasedTraining
from ray.tune.suggest.basic_variant import BasicVariantGenerator

from lfads_torch.run_model import run_model
from lfads_torch.utils import cleanup_best_model, read_pbt_fitlog

logger = logging.getLogger(__name__)

# Get path arguments provided by NeuroCAAS
_, datapath, configpath, resultpath = sys.argv
print(f"--Data: {datapath} Config: {configpath} Results: {resultpath}--")
# Load the YAML file to get PBT parameters
with open(configpath) as yfile:
    cfg = yaml.safe_load(yfile)["pbt"]
# Convert HPs to the correct format
inits, resamps = {}, {}
for n, hp in cfg["hps"].items():
    dist = getattr(tune, hp["dist"])
    inits[n] = dist(*hp["init_range"])
    resamps[n] = dist(*hp["resamp_range"])
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
    config={"datamodule.data_paths.0": datapath, **inits},
    resources_per_trial=cfg["resources_per_trial"],
    num_samples=cfg["num_samples"],
    local_dir=Path(resultpath).parent,
    search_alg=BasicVariantGenerator(random_state=0),
    scheduler=PopulationBasedTraining(
        time_attr="cur_epoch",
        perturbation_interval=cfg["perturbation_interval"],
        burn_in_period=cfg["burn_in_period"],
        hyperparam_mutations=resamps,
        quantile_fraction=cfg["quantile_fraction"],
        resample_probability=cfg["resample_probability"],
        custom_explore_fn=None,
        log_config=True,
        require_attrs=True,
        synch=True,
    ),
    keep_checkpoints_num=1,
    verbose=1,
    progress_reporter=CLIReporter(
        metric_columns=["valid/recon_smth", "cur_epoch"],
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
best_ckpt_dir = best_model_dir / Path(analysis.best_checkpoint.local_path).name
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
cleanup_best_model(best_model_dir)
