# lfads-torch
A PyTorch implementation of Latent Factor Analysis via Dynamical Systems (LFADS).
# Setup
To create an environment and install the dependencies of the project, run the following commands:
```
git clone git@github.com:arsedler9/lfads-torch.git
conda create --name lfads-torch python=3.9
conda activate lfads-torch
cd lfads-torch
pip install -e .
pre-commit install
```

# NotImplemented
- Hyperparameter validation
- Low-dimensional readin layers (and multisession)
- Encoder-only retraining
- Exponentially-smoothed reconstruction metrics
# TODO
## PBT
- Plotting hyperparameter progressions
- PBT stopping criterion

# Known Issues
- A bug within `ray.tune` keeps all trials `PAUSED` when any trial is `TERMINATED` in PBT.
- Using the `burn_in_period` argument to the `PopulationBasedTraining` `scheduler` keeps all trials `PAUSED` after first perturb.
- PTL prints warning messages about restoring from mid-epoch checkpoints, but they should be saved after validation epoch.
- Using `TuneReportCheckpointCallback(..., on='train_end')` fails.
