import logging
import os
import shutil

import ray
from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import FIFOScheduler
from ray.tune.suggest.basic_variant import BasicVariantGenerator

from lfads_torch.run_model import run_model

logger = logging.getLogger(__name__)

# ---------- OPTIONS -----------
LOCAL_MODE = False
OVERWRITE = True
RUN_TAG = "test_new_run"
RUNS_HOME = "/snel/share/runs/lfads-torch/validation"
RUN_DIR = f"{RUNS_HOME}/multi/{RUN_TAG}"
# ------------------------------

# Initialize the `ray` server in local mode if necessary
if LOCAL_MODE:
    ray.init(local_mode=True)
# Overwrite the directory if necessary
if os.path.exists(RUN_DIR):
    if OVERWRITE:
        logger.warning(f"Overwriting multi-run at {RUN_DIR}")
        shutil.rmtree(RUN_DIR)
    else:
        raise OSError(
            "The multi-run directory already exists. "
            "Set `OVERWRITE=True` or create a new `RUN_TAG`."
        )
# Run the hyperparameter search
tune.run(
    tune.with_parameters(run_model, config_name="multi.yaml"),
    metric="valid/recon_smth",
    mode="min",
    name=os.path.basename(RUN_DIR),
    config=dict(
        model=dict(
            dropout_rate=tune.uniform(0.0, 0.7),
            l2_ic_enc_scale=tune.loguniform(1e-5, 1e-3),
            l2_ci_enc_scale=tune.loguniform(1e-5, 1e-3),
            l2_gen_scale=tune.loguniform(1e-5, 1e0),
            l2_con_scale=tune.loguniform(1e-5, 1e0),
            kl_co_scale=tune.loguniform(1e-6, 1e-3),
            kl_ic_scale=tune.loguniform(1e-6, 1e-3),
        ),
    ),
    resources_per_trial=dict(cpu=3, gpu=0.5),
    num_samples=20,
    local_dir=os.path.dirname(RUN_DIR),
    search_alg=BasicVariantGenerator(random_state=0),
    scheduler=FIFOScheduler(),
    verbose=1,
    progress_reporter=CLIReporter(
        metric_columns=["valid/recon_smth", "cur_epoch"],
        sort_by_metric=True,
    ),
    trial_dirname_creator=lambda trial: str(trial),
)
