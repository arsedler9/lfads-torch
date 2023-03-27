import logging
import os
import shutil
from datetime import datetime
from pathlib import Path

from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import PopulationBasedTraining
from ray.tune.suggest.basic_variant import BasicVariantGenerator

from lfads_torch.run_model import run_model

# from ray.tune.stopper import ExperimentPlateauStopper


logger = logging.getLogger(__name__)

# ---------- OPTIONS -----------
OVERWRITE = True
PROJECT_STR = "lfads-torch"
DATASET_STR = "nlb_mc_maze"
RUN_TAG = datetime.now().strftime("%Y%m%d") + "_examplePBT"
RUN_DIR = Path("/snel/share/runs") / PROJECT_STR / DATASET_STR / "pbt" / RUN_TAG

# ------------------------------

# Set the mandatory config overrides to select datamodule and model
mandatory_overrides = {
    "datamodule": DATASET_STR,
    "model": DATASET_STR,
    "logger.wandb_logger.project": PROJECT_STR,
    "logger.wandb_logger.tags.1": DATASET_STR,
    "logger.wandb_logger.tags.2": RUN_TAG,
}
# Overwrite the directory if necessary
if RUN_DIR.exists() and OVERWRITE:
    shutil.rmtree(RUN_DIR)
RUN_DIR.mkdir(parents=True)
# Copy this script into the run directory
shutil.copyfile(__file__, RUN_DIR / Path(__file__).name)
# Run the hyperparameter search
analysis = tune.run(
    tune.with_parameters(
        run_model,
        config_path="../configs/pbt.yaml",
        do_posterior_sample=False,
    ),
    metric="valid/recon_smth",
    mode="min",
    name=RUN_DIR.name,
    # stop=ExperimentPlateauStopper(
    #     metric="valid/recon_smth",
    #     std=1e-5,
    #     top=10,
    #     patience=100,
    # ),
    config={
        **mandatory_overrides,
        "model.lr_init": tune.choice([4e-3]),
        "model.train_aug_stack.transforms.0.cd_rate": tune.choice([0.5]),
        "model.dropout_rate": tune.uniform(0.0, 0.6),
        "model.l2_gen_scale": tune.loguniform(1e-4, 1e0),
        "model.l2_con_scale": tune.loguniform(1e-4, 1e0),
        "model.kl_co_scale": tune.loguniform(1e-6, 1e-4),
        "model.kl_ic_scale": tune.loguniform(1e-5, 1e-3),
    },
    resources_per_trial=dict(cpu=3, gpu=0.5),
    num_samples=20,
    local_dir=RUN_DIR.parent,
    search_alg=BasicVariantGenerator(random_state=0),
    scheduler=PopulationBasedTraining(
        time_attr="cur_epoch",
        perturbation_interval=25,
        burn_in_period=100,  # ramping + perturbation_interval
        hyperparam_mutations={
            "model.lr_init": tune.loguniform(1e-5, 5e-3),
            "model.train_aug_stack.transforms.0.cd_rate": tune.uniform(0.01, 0.7),
            "model.dropout_rate": tune.uniform(0.0, 0.7),
            "model.l2_gen_scale": tune.loguniform(1e-4, 1e0),
            "model.l2_con_scale": tune.loguniform(1e-4, 1e0),
            "model.kl_co_scale": tune.loguniform(1e-6, 1e-4),
            "model.kl_ic_scale": tune.loguniform(1e-5, 1e-3),
        },
        quantile_fraction=0.25,
        resample_probability=0.25,
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
best_model_dir = RUN_DIR / "best_model"
shutil.copytree(analysis.best_logdir, best_model_dir)
# Switch working directory to this folder (usually handled by tune)
os.chdir(best_model_dir)
# Load the best model and run posterior sampling (skip training)
best_ckpt_dir = best_model_dir / Path(analysis.best_checkpoint.local_path).name
run_model(
    checkpoint_dir=best_ckpt_dir,
    config_path="../configs/pbt.yaml",
    do_train=False,
    overrides=mandatory_overrides,
)
