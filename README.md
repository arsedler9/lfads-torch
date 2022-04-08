# lfads-torch
A PyTorch implementation of Latent Factor Analysis via Dynamical Systems (LFADS) and AutoLFADS.
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

# TODO
- Write test suite
- Waiting for next release of `ray` that includes `PopulationBasedTraining` bug fixes
- Waiting on solution to prevent `pytorch_lightning` from restoring previous learning rates during PBT
- Tune default parameters for `ExperimentPlateauStopper`
- Finish applying to all NLB datasets and submit to EvalAI for validation
- (maybe) Define a `tune.Trainable` instead of `run_model` function for computational efficiency

# Known Issues
- Doesn't check hyperparameter values to confirm that they are within the valid ranges.
- PTL prints warning messages about restoring from mid-epoch checkpoints, but they should be saved after validation epoch.
- Using `TuneReportCheckpointCallback(..., on='train_end')` fails.

# Differences from Previous Implementation
There are several known differences from the original implementation. These are mainly out of convenience, and have not been found to affect model performance.
## LFADS Model
- **Read-in layers**: Readin and readout layers, which can be useful for multi-session modeling, haven't been added yet due to implementation hurdles, but will likely be added in the future.
- **Encoder retraining**: This is not implemented as a hyperparameter in `lfads-torch` because we anticipate that such retraining will be more efficiently and flexibly implemented by setting `requires_grad` in individual scripts.
## AutoLFADS
- **Binary tournament**: The binary tournament exploitation strategy has not been added due to implementation challenges. Instead, we use the `PopulationBasedTraining` scheduler from `ray.tune`, which transfers weights from an upper quantile to a lower quantile. HPs are perturbed and, with some smaller probability, resampled.
- **Continuous perturbations**: In the original AutoLFADS implementation, perturbation multipliers were uniformly sampled between 0.8 and 1.2. In the `ray.tune` implementation, perturbations multipliers can take on the discrete values of 0.8 and 1.2.
- **Stopping criterion**: The original criterion for stopping AutoLFADS was based on a percentage improvement threshold, but we use the `ExperimentPlateauStopper` provided with `ray.tune` here for convenience.
