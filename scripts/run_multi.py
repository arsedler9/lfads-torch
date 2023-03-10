import logging
import shutil
from datetime import datetime
from pathlib import Path

from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import FIFOScheduler
from ray.tune.suggest.basic_variant import BasicVariantGenerator

from lfads_torch.run_model import run_model

logger = logging.getLogger(__name__)

# ---------- OPTIONS -----------
OVERWRITE = True
PROJECT_STR = "lfads-torch"
DATASET_STR = "nlb_mc_maze"
RUN_TAG = datetime.now().strftime("%Y%m%d") + "_exampleMulti"
RUN_DIR = Path("/snel/share/runs") / PROJECT_STR / DATASET_STR / "multi" / RUN_TAG
# ------------------------------

# Set the mandatory config overrides to select datamodule and model
mandatory_overrides = {
    "datamodule": DATASET_STR,
    "model": DATASET_STR,
    "logger.wandb_logger.project": PROJECT_STR,
    "logger.wandb_logger.tags.0": RUN_TAG,
}
# Overwrite the directory if necessary
if RUN_DIR.exists() and OVERWRITE:
    shutil.rmtree(RUN_DIR)
RUN_DIR.mkdir(parents=True)
# Copy this script into the run directory
shutil.copyfile(__file__, RUN_DIR / Path(__file__).name)
# Run the hyperparameter search
tune.run(
    tune.with_parameters(
        run_model,
        config_path="../configs/multi.yaml",
    ),
    metric="valid/recon_smth",
    mode="min",
    name=RUN_DIR.name,
    config={
        **mandatory_overrides,
        "model.dropout_rate": tune.uniform(0.0, 0.15),
        "model.l2_ic_enc_scale": tune.loguniform(1e-5, 1e-3),
        "model.l2_ci_enc_scale": tune.loguniform(1e-5, 1e-3),
        "model.l2_gen_scale": tune.loguniform(1e-5, 1e0),
        "model.l2_con_scale": tune.loguniform(1e-5, 1e0),
        "model.kl_co_scale": tune.loguniform(1e-6, 1e-3),
        "model.kl_ic_scale": tune.loguniform(1e-6, 1e-3),
    },
    resources_per_trial=dict(cpu=2, gpu=0.3),
    num_samples=90,
    local_dir=RUN_DIR.parent,
    search_alg=BasicVariantGenerator(random_state=0),
    scheduler=FIFOScheduler(),
    verbose=1,
    progress_reporter=CLIReporter(
        metric_columns=["valid/recon_smth", "cur_epoch"],
        sort_by_metric=True,
    ),
    trial_dirname_creator=lambda trial: str(trial),
)
