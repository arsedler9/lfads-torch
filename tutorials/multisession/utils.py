import glob
import os

import h5py
import matplotlib.cm as colormap
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal
from scipy.io import loadmat
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold


def get_global_pcs(trial_averaged_all, channel_mean_all, n_pcs=100):
    """
    Get PCs of all datasets simultaneously.
    Parameters
    -------
    tr_averaged_all: np.array
        Concatenated trial averaged data of shape (nbins * nconds) x (ndatasets * nchans)
    channel_mean_all: np.array
        Per-channel means of tr_averaged_all, of shape (ndatasets x nchans)
    n_pcs: int
        Number of principal components to keep, by default 100.
    Returns
    -------
    dim_reduced_data: np.array
        Global principal components, of shape (nbins * nconds) x npcs
    """

    # mean-center
    n_channels_total = channel_mean_all.shape[0]
    tr_avg_mean_centered = np.array(
        [
            trial_averaged_all[:, ii] - channel_mean_all[ii]
            for ii in range(n_channels_total)
        ]
    ).T
    # check that new data is close to mean 0
    assert all(np.mean(tr_avg_mean_centered, axis=0) < 1e-10)

    pca = PCA(n_components=n_pcs)
    pca.fit(tr_avg_mean_centered)

    # after running pca, pca.components_.shape should be [pcs_to_keep x nchannels]
    assert pca.components_.shape == (n_pcs, trial_averaged_all.shape[1])

    # project the trial-averaged data through the PCA projection
    dim_reduced_data = np.dot(tr_avg_mean_centered, pca.components_.T)
    # should now have shape [(ntimesteps*nconditions) x pcs_to_keep]
    assert dim_reduced_data.shape == (trial_averaged_all.shape[0], n_pcs)

    return dim_reduced_data, pca


def run_pcr(tr_averaged_days, channel_mean_days, dim_reduced_data, ds_names):
    """
    Run PCR to map single day data to global PCs
    Parameters
    -------
    tr_averaged_days: list of np.arrays
        List of trial averaged data for each day of shape ndatasets x (nbins * nconds) x nchans
    channel_mean_days: list of np.array
        Per-channel means of tr_averaged_all, of shape ndatasets x nchans
    dim_reduced_data: np.array
        Global principal components, of shape (nbins * nconds) x npcs
    ds_names: list of strings
        List of dataset names to be used as keys in output dicts
    Returns
    -------
    alignment_matrices : dictionary of np.arrays
        Dictionary containing learned weight matrix for each dataset
    alignment_biases : dictionary of np.arrays
        Dictionary containing projected means for each dataset
    """

    alignment_matrices = {}
    alignment_biases = {}
    latents = {}

    pcs_to_keep = dim_reduced_data.shape[1]
    for iday, data in enumerate(tr_averaged_days):
        reg = Ridge(alpha=1.0, fit_intercept=False)
        # regress the mean centered single-day data onto the dim reduced data
        data_mean_centered = data - channel_mean_days[iday]
        # data_mean_centered = data
        # just make sure we properly mean centered
        assert all(np.mean(data_mean_centered, axis=0) < 1e-10)

        reg.fit(data_mean_centered, dim_reduced_data)
        latents[ds_names[iday]] = reg.predict(data_mean_centered)
        # after regression, reg.coef_ should be [pcs_to_keep x nchannels]
        assert reg.coef_.shape == (pcs_to_keep, tr_averaged_days[iday].shape[1])

        W = reg.coef_.T
        alignment_matrices[ds_names[iday]] = W  # nchans x nPCs
        alignment_biases[ds_names[iday]] = channel_mean_days[iday]  # nchans

    return alignment_matrices, alignment_biases, latents


def get_data_from_PSTH_prep(datapath, field, bin_size_s=0.01):
    """
    Load a variable from the PSTH_prep.mat files
    Parameters
    -------
    datapath: str
        String of path to directory containing PSTH_prep files
    field: str
        Name of variable you want to load
    bin_size_s: float
        Bin size in seconds, for correcting spike counts (default = 10 ms bins)
    Returns
    -------
    all_data: dict
        Dictionary mapping dataset date to numpy array containing variable's data
        for that dataset.
    """
    datasets = sorted(glob.glob(os.path.join(datapath, "*PSTH_prep*.mat")))

    # somewhere to store the data
    all_data = {}

    for ds in datasets:
        # get date - use as identifier
        date = ds.split("/")[-1].split("_")[2].split("-")[0]
        # load dataset
        data = loadmat(ds)
        # this is the variable being averaged in line 131, PSTH_prep.m
        data_field = data[field]

        if "spike" in field:
            data_field = data_field * bin_size_s

        all_data[date] = data_field

    return all_data


def smooth_spikes(aligned_spikes, gaussian_std_s, dt, valid=True, nanpad=True):
    """
    Smooth input spiking data.
    Parameters
    -------
    aligned_spikes: dict
        Dictionary mapping dataset date to numpy array containing spikes
        spikes to be smoothed for that dataset
    gaussian_std_s: float
        Standard deviation of the gaussian to use for smoothing in seconds.
    dt: float
        Bin width in seconds.
    Returns
    -------
    smoothed_aligned_spikes: dict
        Dictionary mapping dataset date to numpy array containing smoothed
        spikes for that dataset
    """
    # Set up a Gaussian window for smoothing
    # need to scale by our timestep
    gaussian_std_steps = gaussian_std_s / dt

    # we need a window length of ~3 standard deviations on each side (x2)
    # in order for the Gaussian to approach 0 on each side
    window_length = gaussian_std_steps * 3 * 2

    # use scipy to get the gaussian
    gaussian_window = signal.gaussian(M=window_length, std=gaussian_std_steps, sym=True)

    # normalize the signal so it has a "mass" of 1
    # (this is important for convolution)
    gaussian_window = gaussian_window / np.sum(gaussian_window)

    # length of invalid stuff to remove after filtering
    shift_len = len(gaussian_window) // 2

    # hold smoothed trials
    smoothed_aligned_spikes = {}
    # for each dataset
    for k in aligned_spikes:
        data = aligned_spikes[k]
        neurons, time, trials = data.shape

        if not nanpad:
            time -= shift_len

        smoothed_spikes = np.zeros((neurons, time, trials))
        # for each trial
        for tr in range(trials):
            # for each neuron
            for ne in range(neurons):
                noisy_data = data[ne, :, tr]
                convolved_fr = signal.lfilter(gaussian_window, 1, noisy_data)
                # convolved_fr = convolved_fr * 1./dt # NOTE: unnecessary
                if valid:
                    if nanpad:
                        convolved_fr_valid = np.concatenate(
                            [
                                np.full(int((window_length) // 2), np.nan),
                                convolved_fr[int(window_length) :],
                                np.full(int((window_length) // 2), np.nan),
                            ],
                            axis=0,
                        )
                    else:
                        convolved_fr_valid = convolved_fr[shift_len:]
                else:
                    convolved_fr_valid = convolved_fr
                smoothed_spikes[ne, :, tr] = convolved_fr_valid

        smoothed_aligned_spikes[k] = smoothed_spikes

    return smoothed_aligned_spikes


def plot_firing_rates(
    firing_rates,
    conditionIDs,
    title,
    neurons=None,
    single_trial=False,
    n_single_trials=10,
    lims=None,
    scale=None,
):
    """
    Creates as many 5 x 4 grids of plots as is necessary to plot all neurons in
    a given dataset. Can be used to plot either PSTHs or single trial firing rates.
    Creates ceil(num_neurons/20) plot for each key in rates_dict.
    Parameters
    -------
    firing_rates: dict
        Dictionary of string keys (expected to be dates of datasets) to numpy arrays
        containing rate variable. Commonly smoothed spikes or lfads rates.
    conditionIDs: dict
       Dictionary of string keys (dates - should be the same as neural_data) to numpy
       arrays containing condition ID information from targetID_peakVel variable.
    single_trial: bool
        Plots single trial rates if True. Computes and plots PSTHs if False.
    n_single_trials: int
        Number of single trials to plot if single_trials is True. Not used if single_trials
        is False.
    Returns
    -------
    None. Displays ceil(num_neurons/20) plot for each key in the dictionaries.
    """
    # if the scale is none, multiply by 1 to keep constant
    if scale == None:
        scale = 1.0
    # preprocess conditions so there are only 8
    for c in conditionIDs:
        if c > 24:
            c -= 24
        elif c > 16:
            c -= 16
        elif c > 8:
            c -= 8

    n_conds = len(np.unique(conditionIDs))
    if neurons is None:
        n_neurons = firing_rates.shape[0]
    else:
        n_neurons = len(neurons)
        firing_rates = firing_rates[neurons, :, :]
    n_time = firing_rates.shape[1]
    n_plots = int(np.ceil(n_neurons / 20.0))

    axs = []
    for _ in range(n_plots):
        fig, ax = plt.subplots(5, 4, figsize=(8, 10))
        plt.subplots_adjust(wspace=0.5, hspace=0.5)
        axs.append(ax.flatten())
    axs = np.array(axs).flatten()

    cm = colormap.hsv

    if single_trial:
        for c in range(n_conds):
            clr = cm(c / float(n_conds))

            where_cond = np.where(conditionIDs == c + 1)[0]
            cond_trials = firing_rates[:, :, where_cond]
            for n in range(n_neurons):
                axs[n].plot(
                    cond_trials[n, :, 0:n_single_trials] / scale,
                    color=clr,
                    linewidth=0.75,
                    alpha=0.6,
                )
                axs[n].set_title("Neuron " + str(n))
                if lims:
                    axs[n].set_xlim(lims)

            plt.suptitle("Single Trial - " + title)
    else:
        condition_averaged_rates = np.full((n_conds, n_neurons, n_time), np.nan)

        for cond in np.unique(conditionIDs):
            where_cond = np.where(conditionIDs == cond)[0]
            cond_trials = firing_rates[:, :, where_cond]
            cond_avg = np.mean(cond_trials, axis=-1)
            condition_averaged_rates[cond - 1, :, :] = cond_avg

        for c in range(n_conds):
            clr = cm(c / float(n_conds))
            for n in range(n_neurons):
                axs[n].plot(condition_averaged_rates[c, n, :] / scale, color=clr)
                axs[n].set_title("Neuron " + str(n))
                if lims:
                    axs[n].set_xlim(lims)
            plt.suptitle("Condition Averaged - " + title)

    plt.show(block=False)


def get_lfads_output(output_files, binwidth):
    lfads_factors = {}
    lfads_rates = {}
    for file in output_files:
        session = file.split("_")[-1].split(".")[0]
        h5_out = h5py.File(file, "r")
        factors = np.zeros(
            (
                h5_out["train_inds"].shape[0] + h5_out["valid_inds"].shape[0],
                h5_out["train_factors"].shape[1],
                h5_out["train_factors"].shape[2],
            )
        )
        rates = np.zeros(
            (
                h5_out["train_inds"].shape[0] + h5_out["valid_inds"].shape[0],
                h5_out["train_output_params"].shape[1],
                h5_out["train_output_params"].shape[2],
            )
        )
        factors[h5_out["train_inds"], :, :] = h5_out["train_factors"]
        factors[h5_out["valid_inds"], :, :] = h5_out["valid_factors"]
        rates[h5_out["train_inds"], :, :] = h5_out["train_output_params"]
        rates[h5_out["valid_inds"], :, :] = h5_out["valid_output_params"]
        lfads_factors[session] = np.swapaxes(factors, 0, 2)
        lfads_rates[session] = np.swapaxes(rates, 0, 2) / binwidth

    return lfads_rates, lfads_factors


def compute_central_difference(signal):
    """
    Approximate the derivative of a signal with numpy.gradient, which uses
    second order accurate central differences to perform the calculation.
    Parameters
    ----------
    signal : np.array
       trial data in shape dims x time x trials
    Returns
    -------
    signal_diff : np.array
        same format as input, but all trials are the derivative of the original
        signal
    """
    signal_diff = np.zeros_like(signal)
    n_trials = signal.shape[-1]
    for ii in range(n_trials):
        signal_diff[:, :, ii] = np.array(np.gradient(signal[:, :, ii], axis=1))

    return signal_diff


def run_five_fold_cv_decoding(neural_data, behavioral_data, n_lag=0):
    if n_lag > 0:
        # trim the end of the predictor
        stacked_days_neural_data = np.concatenate(
            [neural_data[k][:, :-n_lag, :] for k in sorted(neural_data.keys())], axis=-1
        )
        # trim the beginning of the predictee
        stacked_days_behavior = np.concatenate(
            [behavioral_data[k][:, n_lag:, :] for k in sorted(behavioral_data.keys())],
            axis=-1,
        )
    else:
        stacked_days_neural_data = np.concatenate(
            [neural_data[k] for k in sorted(neural_data.keys())], axis=-1
        )
        stacked_days_behavior = np.concatenate(
            [behavioral_data[k] for k in sorted(behavioral_data.keys())], axis=-1
        )
    # stack trials into 2D arrays for decoding
    X = np.vstack(np.swapaxes(stacked_days_neural_data, 2, 0))
    y = np.vstack(np.swapaxes(stacked_days_behavior, 0, 2))
    # should only apply for smoothed spikes
    nan_inds = ~np.isnan(X[:, 0])
    X_nan = X[nan_inds, :]
    y_nan = y[nan_inds, :]

    # split data into five folds
    kf = KFold(n_splits=5, shuffle=True)

    perfs = []
    decoders = []
    for train_index, test_index in kf.split(X_nan, y_nan):
        train_X = X_nan[train_index, :]
        test_X = X_nan[test_index, :]
        train_y = y_nan[train_index, :]
        test_y = y_nan[test_index, :]

        decoder = Ridge()  # linear decoder with L2 regularization
        # fit the decoder
        decoder.fit(train_X, train_y)
        score = decoder.score(test_X, test_y)
        perfs.append(score)
        decoders.append(decoder)

    best_decoder = decoders[np.argmax(perfs)]

    return np.mean(perfs), best_decoder


def evaluate_decoder(decoder, neural_data, behavioral_data, n_lag=0):
    """
    Evaluate trained Ridge regression decoder on all pairs of neural and behavioral
    data provided in the dictionaries.
    Parameters
    -------
    decoder: sklearn.linear_model.Ridge
        Trained decoder object from fit_decoder function.
    neural_data: dict
        Dictionary of string keys (expected to be dates of datasets) to numpy arrays
        containing desired neural variable (e.g. lfads factors) to decode from
    behavioral_data: dict
       Dictionary of string keys (dates - should be the same as neural_data) to numpy
       arrays containing desired behavioral variable (e.g. joystick position) to decode to
    Returns
    -------
    preds: dict
        Dictionary of string keys (will match those in neural_data) to numpy arrays
        containing predicted behavioral data.
    perf: dict
        Dictionary of string keys (will match those in neural_data) to float values
        representing the R^2 score of the predictions vs. the actual behavior
    """
    preds = {}
    perf = {}
    for k in neural_data:
        if n_lag > 0:
            # trim the end of the predictor
            neur = neural_data[k][:, :-n_lag, :]
            # trim the beginning of the predictee
            behav = behavioral_data[k][:, n_lag:, :]
        else:
            neur = neural_data[k]
            behav = behavioral_data[k]

        X = np.vstack(np.swapaxes(neur, 2, 0))
        y = np.vstack(np.swapaxes(behav, 0, 2))

        nan_inds = ~np.isnan(X[:, 0])
        X_nan = X[nan_inds, :]
        y_nan = y[nan_inds, :]

        y_hat = decoder.predict(X_nan)
        score = decoder.score(X_nan, y_nan)

        predictions = np.full(y.shape, np.nan)
        predictions[nan_inds, :] = y_hat

        preds[k] = predictions
        perf[k] = score

    return preds, perf
