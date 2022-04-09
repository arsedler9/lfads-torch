import logging
import os
import shutil

from lfads_torch.run_model import run_model

logger = logging.getLogger(__name__)

# ---------- OPTIONS -----------
OVERWRITE = True
RUN_TAG = "test_new_run"
RUNS_HOME = "/snel/share/runs/lfads-torch/validation"
RUN_DIR = f"{RUNS_HOME}/single/{RUN_TAG}"
# ------------------------------

# Overwrite the directory if necessary
if os.path.exists(RUN_DIR):
    if OVERWRITE:
        logger.warning(f"Overwriting single-run at {RUN_DIR}")
        shutil.rmtree(RUN_DIR)
    else:
        raise OSError(
            "The single-run directory already exists. "
            "Set `OVERWRITE=True` or create a new `RUN_TAG`."
        )
# Switch to the `RUN_DIR` and train the model
os.makedirs(RUN_DIR)
os.chdir(RUN_DIR)
run_model(config_name="single.yaml")
