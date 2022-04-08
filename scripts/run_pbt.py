import logging
import os
import shutil

import ray
from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import PopulationBasedTraining
from ray.tune.suggest.basic_variant import BasicVariantGenerator

from lfads_torch.run_model import run_model

# from ray.tune.stopper import ExperimentPlateauStopper


logger = logging.getLogger(__name__)

# ---------- OPTIONS -----------
LOCAL_MODE = False
OVERWRITE = True

RUN_TAG = "test_new_run"
RUNS_HOME = "/snel/share/runs/lfads-torch/validation"
# ------------------------------

# Initialize the `ray` server in local mode if necessary
if LOCAL_MODE:
    ray.init(local_mode=True)
# Overwrite the directory if necessary
RUN_DIR = f"{RUNS_HOME}/pbt/{RUN_TAG}"
if os.path.exists(RUN_DIR):
    if OVERWRITE:
        logger.warning(f"Overwriting pbt run at {RUN_DIR}")
        shutil.rmtree(RUN_DIR)
    else:
        raise OSError(
            "The pbt run directory already exists. "
            "Set `OVERWRITE=True` or create a new `RUN_TAG`."
        )
# Run the hyperparameter search
analysis = tune.run(
    tune.with_parameters(
        run_model,
        config_train="pbt.yaml",
        do_posterior_sample=False,
    ),
    metric="valid/recon_smth",
    mode="min",
    name=RUN_TAG,
    # stop=ExperimentPlateauStopper(
    #     metric="valid/recon_smth",
    #     std=1e-5,
    #     top=10,
    #     patience=100,
    # ),
    config=dict(
        model=dict(
            lr_init=tune.choice([1e-2]),
            cd_rate=tune.choice([0.5]),
            dropout_rate=tune.uniform(0.0, 0.6),
            l2_gen_scale=tune.loguniform(1e-4, 1e0),
            l2_con_scale=tune.loguniform(1e-4, 1e0),
            kl_co_scale=tune.loguniform(1e-6, 1e-4),
            kl_ic_scale=tune.loguniform(1e-5, 1e-3),
        ),
    ),
    resources_per_trial=dict(cpu=3, gpu=0.5),
    num_samples=20,
    local_dir=f"{RUNS_HOME}/pbt",
    search_alg=BasicVariantGenerator(random_state=0),
    scheduler=PopulationBasedTraining(
        time_attr="cur_epoch",
        perturbation_interval=70,  # 50
        # burn_in_period=150,
        hyperparam_mutations=dict(
            model=dict(
                lr_init=tune.loguniform(1e-5, 5e-3),
                cd_rate=tune.uniform(0.01, 0.7),
                dropout_rate=tune.uniform(0.0, 0.6),
                l2_gen_scale=tune.loguniform(1e-4, 1e0),
                l2_con_scale=tune.loguniform(1e-4, 1e0),
                kl_co_scale=tune.loguniform(1e-6, 1e-4),
                kl_ic_scale=tune.loguniform(1e-5, 1e-3),
            ),
        ),
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
    reuse_actors=True,
)
# Load the best model and run posterior sampling (skip training)
best_model_dir = os.path.join(os.getcwd(), "best_model")
shutil.copytree(analysis.best_logdir, best_model_dir)
run_model(
    # TODO: Update to use `ray.tune` checkpoints (analysis.best_checkpoint)
    checkpoint_dir=os.path.join(best_model_dir, "ptl_ckpts"),
    config_train="pbt.yaml",
    do_train=False,
)
