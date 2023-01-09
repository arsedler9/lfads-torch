import logging
import os
import shutil
from datetime import datetime
from pathlib import Path

from lfads_torch.run_model import run_model

logger = logging.getLogger(__name__)

# ---------- OPTIONS -----------
OVERWRITE = True
RUN_TAG = datetime.now().strftime("%Y%m%d-%H%M%S")
RUNS_HOME = Path("/snel/share/runs/lfads-torch/validation")
RUN_DIR = RUNS_HOME / "single" / RUN_TAG
CONFIG_PATH = Path("../configs/single.yaml")
# ------------------------------

# Overwrite the directory if necessary
if RUN_DIR.exists() and OVERWRITE:
    shutil.rmtree(RUN_DIR)
RUN_DIR.mkdir(parents=True)
# Switch to the `RUN_DIR` and train the model
os.chdir(RUN_DIR)
run_model(
    overrides={
        "datamodule": "nlb_mc_maze",
        "model": "nlb_mc_maze",
    },
    config_path=CONFIG_PATH,
)
