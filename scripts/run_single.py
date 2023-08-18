import os
import sys

from lfads_torch.run_model import run_model

# Get path arguments provided by NeuroCAAS
_, datapath, configpath, resultpath = sys.argv
print(f"--Data: {datapath} Config: {configpath} Results: {resultpath}--")
# Run a single LFADS model
os.chdir(resultpath)
run_model(
    overrides={"datamodule.data_paths.0": datapath},
    config_path=configpath,
)
