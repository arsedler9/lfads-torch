# Short Description:

LFADS with automated hyperparameter optimization.

# Long Description:

[Latent factor analysis via dynamical systems (LFADS)](https://www.nature.com/articles/s41592-018-0109-9) is a variational sequential autoencoder that achieves state-of-the-art performance in denoising high-dimensional neural spiking activity for downstream applications in science and engineering. Recently introduced variants have continued to demonstrate the applicability of the architecture to a wide variety of problems in neuroscience.

Since the development of the original implementation of LFADS, new technologies have emerged that use [dynamic computation graphs](https://pytorch.org/), [minimize boilerplate code](https://www.pytorchlightning.ai/index.html), [compose model configuration files](https://hydra.cc/), and [simplify large-scale training](https://docs.ray.io/en/latest/tune/index.html). This implementation of LFADS, [`lfads-torch`](https://github.com/arsedler9/lfads-torch/tree/neurocaas), builds on these modern Python libraries and is designed to be easier to understand, configure, and extend.

Achieving state-of-the-art performance with deep neural population dynamics models requires extensive hyperparameter tuning for each dataset. [AutoLFADS](https://www.nature.com/articles/s41592-022-01675-0) is a model-tuning framework that automatically produces high-performing LFADS models on data from a variety of brain areas and tasks, without behavioral or task information. The AutoLFADS framework uses [coordinated dropout](https://arxiv.org/abs/1908.07896) to prevent identity overfitting and [population-based training](https://arxiv.org/abs/1711.09846) to efficiently tune hyperparameters over the course of a single training run.

# Paper Link:

https://www.nature.com/articles/s41592-022-01675-0

[TODO: Add lfads-torch preprint when available]

# Github Repo Link:

https://github.com/arsedler9/lfads-torch/tree/neurocaas

# Bash Script Link:

https://github.com/arsedler9/lfads-torch/blob/neurocaas/run_main.sh

# Demo Link:

https://drive.google.com/drive/folders/1eOxuPsGVes_1FUemsHWQHZ_OccQuj184

# How to use this analysis:

NeuroCAAS runs the `neurocaas` branch of `lfads-torch`, which is publicly available at the link above. At a high level, the pipeline takes (1) an input data file in the HDF5 format along with (2) a configuration YAML file which specifies model architecture and training hyperparameters. At the end of training, the pipeline will return a `.zip` file which contains training logs, model outputs, and the best performing model checkpoint.

## Data File

At a minimum, the HDF5 data file must have the following keys, in the `n_samples x n_timesteps x n_channels` format:

- `train_encod_data`: Data to be used as input when training the model.
- `train_recon_data`: Data to be used as a reconstruction target when training the model.
- `valid_encod_data`: Data to be used as input when validating the model.
- `valid_recon_data`: Data to be used as a reconstruction target when validating the model.

## Configuration File

### Basic Configuration

Starting from the demo config file, you'll need to edit the following fields to get the code to run:

- `encod_data_dim`: The `n_channels` dimension of `encod_data` from your data file.
- `encod_seq_len`: The `n_timesteps` dimension of `encod_data` from your data file.
- `recon_seq_len`: The `n_timesteps` dimension of `recon_data` from your data file.
- `readout.modules.0.out_features`: The `n_channels` dimension of `recon_data` from your data file.
- `datamodule.batch_size`: The batch size to use for training and inference. Should mainly be used to manage memory usage. Two models per GPU is typical.
- `datamodule.batch_keys` and `datamodule.attr_keys`: Only relevant for computing auxiliary metrics within custom callbacks (e.g., for the Neural Latents Benchmark). Can be removed in most cases.

The following hyperparameters are also frequently changed, in order of importance:

- `pbt.hps`: The ranges from which PBT should initialize and resample learning rates and regularization hyperparameters.
- `trainer.max_epochs`: The total number of epochs to train the model.
- `pbt.perturbation_interval`: The number of epochs between successive perturbations of the population.
- `pbt.burn_in_period`: The number of epochs to wait before the first hyperparameter perturbation.
- `model.gen_dim`: The size of the generator RNN.
- `model.fac_dim`: The size of the latent factors dimensionality bottleneck.

### Advanced Configuration

The configuration system for `lfads-torch` is based on Hydra. [Under the hood](https://github.com/arsedler9/lfads-torch/blob/neurocaas/lfads_torch/run_model.py), the `model`, `datamodule` and `trainer` keys are recursively instantiated by passing the arguments specified to their respective `_target_` objects. For example, in order for Hydra to instantiate the `model`, it first instantiates the `readin`, `readout`, `train_aug_stack`, `infer_aug_stack`, etc. and then passes these objects into the `LFADS` constructor. This is important to understand this in order to take full advantage of the modularity of `lfads-torch`. Augmentations, priors, and reconstruction costs can all be easily swapped out in this way to develop new functionality. See [the GitHub repo](https://github.com/arsedler9/lfads-torch/blob/neurocaas) for more detail.

## Output Files

A successfully completed AutoLFADS run will return an `autolfads.zip` file containing the following:

- `fitlog.csv`: A log of the many metrics reported during training. The most important metrics to pay attention to are the reconstruction losses, `train/recon` and `valid/recon`. For most established model variants, this refers to the mean negative log-likelihood. Other metrics are clearly labeled in the header of the CSV. We recommend inspection of `train/recon` and `valid/recon` to avoid common issues like underfitting and overfitting.
- `lfads_output_sess0.h5`: A copy of the original data file, combined with the outputs of the trained model. The denoised trials can be found at `train_output_params` and `valid_output_params`, along with other intermediate latent representations.
- `model.ckpt`: The parameters of the final trained model. The entire model can be loaded for post-hoc inspection and inference by passing the path to this checkpoint into the `LFADS.load_from_checkpoint` function.
