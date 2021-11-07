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
- Hyperparameter updates for PBT
- Low-dimensional readin layers (and multisession)
- Autoregressive controller output prior with Monte Carlo KL (+ others?)
- Encoder-only retraining
- Exponentially-smoothed reconstruction metrics
# TODO
- Most recent and best reconstruction checkpoints
