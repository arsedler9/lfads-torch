import logging
import shutil
from glob import glob
from pathlib import Path
import pickle
import h5py
import torch
from tqdm import tqdm
import numpy as np
import pandas as pd
import json
import snel_toolkit.decoding as dec
from snel_toolkit.analysis import PSTH
import sklearn
from scipy.special import gammaln
from ..datamodules import reshuffle_train_valid
from ..tuples import SessionOutput
from ..utils import send_batch_to_device, transpose_lists

logger = logging.getLogger(__name__)

def neg_log_likelihood(rates, spikes, zero_warning=True):
    """Calculates Poisson negative log likelihood given rates and spikes.
    formula: -log(e^(-r) / n! * r^n)
           = r - n*log(r) + log(n!)
    
    Parameters
    ----------
    rates : np.ndarray
        numpy array containing rate predictions
    spikes : np.ndarray
        numpy array containing true spike counts
    zero_warning : bool, optional
        Whether to print out warning about 0 rate 
        predictions or not
    
    Returns
    -------
    float
        Total negative log-likelihood of the data
    """
    assert spikes.shape == rates.shape, \
        f"neg_log_likelihood: Rates and spikes should be of the same shape. spikes: {spikes.shape}, rates: {rates.shape}"

    if np.any(np.isnan(spikes)):
        mask = np.isnan(spikes)
        rates = rates[~mask]
        spikes = spikes[~mask]
    
    assert not np.any(np.isnan(rates)), \
        "neg_log_likelihood: NaN rate predictions found"

    assert np.all(rates >= 0), \
        "neg_log_likelihood: Negative rate predictions found"
    if (np.any(rates == 0)):
        if zero_warning:
            print("neg_log_likelihood: Zero rate predictions found. Replacing zeros with 1e-9")
        rates[rates == 0] = 1e-9
    
    result = rates - spikes * np.log(rates) + gammaln(spikes + 1.0)
    return np.sum(result)

def bits_per_spike(rates, spikes):
    """Computes bits per spike of rate predictions given spikes.
    Bits per spike is equal to the difference between the log-likelihoods (in base 2)
    of the rate predictions and the null model (i.e. predicting mean firing rate of each neuron)
    divided by the total number of spikes.

    Parameters
    ----------
    rates : np.ndarray
        3d numpy array containing rate predictions
    spikes : np.ndarray
        3d numpy array containing true spike counts
    
    Returns
    -------
    float
        Bits per spike of rate predictions
    """
    nll_model = neg_log_likelihood(rates, spikes)
    nll_null = neg_log_likelihood(np.tile(np.nanmean(spikes, axis=(0,1), keepdims=True), (spikes.shape[0], spikes.shape[1], 1)), spikes, zero_warning=False)
    return (nll_null - nll_model) / np.nansum(spikes) / np.log(2)

def run_posterior_sampling(model, datamodule, filename, num_samples=50):
    """Runs the model repeatedly to generate outputs for different samples
    of the posteriors. Averages these outputs and saves them to an output file.

    Parameters
    ----------
    model : lfads_torch.model.LFADS
        A trained LFADS model.
    datamodule : pytorch_lightning.LightningDataModule
        The `LightningDataModule` to pass through the `model`.
    filename : str
        The filename to use for saving output
    num_samples : int, optional
        The number of forward passes to average, by default 50
    """
    # Convert filename to pathlib.Path for convenience
    filename = Path(filename)
    # Set up the dataloaders
    datamodule.setup()
    pred_dls = datamodule.predict_dataloader()
    # Set the model to evaluation mode
    model.eval()

    # Function to run posterior sampling for a single session at a time
    def run_ps_batch(s, batch):
        # Move the batch to the model device
        batch = send_batch_to_device({s: batch}, model.device)
        # Repeatedly compute the model outputs for this batch
        for i in range(num_samples):
            # Perform the forward pass through the model
            output = model.predict_step(batch, None, sample_posteriors=True)[s]
            # Use running sum to save memory while averaging
            if i == 0:
                # Detach output from the graph to save memory on gradients
                sums = [o.detach() for o in output]
            else:
                sums = [s + o.detach() for s, o in zip(sums, output)]
        # Finish averaging by dividing by the total number of samples
        return [s / num_samples for s in sums]

    # Compute outputs for one session at a time
    for s, dataloaders in pred_dls.items():
        # Copy data file for easy access to original data and indices
        dhps = datamodule.hparams
        data_paths = sorted(glob(dhps.datafile_pattern))
        # Give each session a unique file path
        session = data_paths[s].split("/")[-1].split(".")[0]
        sess_fname = f"{filename.stem}_{session}{filename.suffix}"
        if dhps.reshuffle_tv_seed is not None:
            # If the data was shuffled, shuffle it when copying
            with h5py.File(data_paths[s]) as h5file:
                data_dict = {k: v[()] for k, v in h5file.items()}
            data_dict = reshuffle_train_valid(
                data_dict, dhps.reshuffle_tv_seed, dhps.reshuffle_tv_ratio
            )
            with h5py.File(sess_fname, "w") as h5file:
                for k, v in data_dict.items():
                    h5file.create_dataset(k, data=v)
        else:
            shutil.copyfile(data_paths[s], sess_fname)
        for split in dataloaders.keys():
            # Compute average model outputs for each session and then recombine batches
            logger.info(f"Running posterior sampling on Session {s} {split} data.")
            with torch.no_grad():
                post_means = [
                    run_ps_batch(s, batch) for batch in tqdm(dataloaders[split])
                ]
            post_means = SessionOutput(
                *[torch.cat(o).cpu().numpy() for o in transpose_lists(post_means)]
            )
            # Save the averages to the output file
            with h5py.File(sess_fname, mode="a") as h5file:
                for name in SessionOutput._fields:
                    h5file.create_dataset(
                        f"{split}_{name}", data=getattr(post_means, name)
                    )
        # Log message about sucessful completion
        logger.info(f"Session {s} posterior means successfully saved to `{sess_fname}`")

def run_post_evaluation(model, datamodule, filename):

    datamodule.setup()
    dhps = datamodule.hparams
    data_paths = sorted(glob(dhps.datafile_pattern))
    filename = Path(filename)
    
    for path in data_paths:
        data_path = Path(path)
        interface_path = glob(str(data_path.parent)+'/*_interface.pkl')[0]
        raw_data_path = glob(str(data_path.parent)+'/*_raw.pkl')[0]
        with open(interface_path, 'rb') as f:
            interface = pickle.load(f)
        with open(raw_data_path, 'rb') as f:
            ds = pd.read_pickle(f)
    
        n_segments = sum([sr.n_chops for sr in interface.segment_records])

        # load previous output drop inds
        with h5py.File(data_path, 'r') as h5f:
            seg_idx = h5f['segment_inds'][()]
            heldout_idx = h5f['heldout_inds'][()]
            heldin_idx = h5f['heldin_inds'][()]
            valid_heldout_truth = h5f['valid_recon_data'][:,3:-3,heldout_idx]

        assert np.max(seg_idx) < n_segments
        sess_fname = f"{filename.stem}_{data_path.stem}{filename.suffix}"
        # load lfads output
        with h5py.File(sess_fname, 'r') as h5f:
            rates = []
            factors = []
            inds = []
            valid_heldout_pred = h5f[f'valid_output_params'][:,3:-3,heldout_idx]
            for split in ['train', 'valid']:
                rates.append(h5f[f'{split}_output_params'][()])
                factors.append(h5f[f'{split}_factors'][()])
                inds.append(h5f[f'{split}_inds'][()])

        # merging time
        rates_arr = np.full((n_segments, rates[0].shape[1]-6, rates[0].shape[2]), np.nan)
        factors_arr = np.full((n_segments, factors[0].shape[1]-6, factors[0].shape[2]), np.nan)
        
        for s_rates, s_factors, s_inds in zip(rates, factors, inds):
            true_s_inds = seg_idx[s_inds]
            rates_arr[true_s_inds,:,:] = s_rates[:,3:-3,:]
            factors_arr[true_s_inds,:,:] = s_factors[:,3:-3,:]

        data_dict = {
            'rates': rates_arr,
            'factors': factors_arr,
        }

        merged_df = interface.merge(data_dict)
        ds.data = pd.concat([ds.data, merged_df], axis=1)
        n_factors = factors[0].shape[2]
        facs = ["%04d" % x for x in range(n_factors)]
        cols = [('lfads_factors', fac) for fac in facs]
        channels = ds.data.spikes.columns.values
        cols += [('spikes', ch) for ch in channels]

        ds.data = ds.data.dropna(subset=cols)

        # smooth the spikes
        gauss_width = 45 #ms
        ds.smooth_spk(gauss_width,
                    name="smooth")

        # calculate the target angles
        rel_tgt_angle = np.round(
            ds.trial_info[["target_pos_x","target_pos_y"]].diff()
            .dropna()
            .apply(
                lambda x: np.mod(
                    ((np.math.atan2(x.target_pos_y, x.target_pos_x) * 360) / (2 * np.pi) + 180), 360
                ), axis=1
            )
        )
        ds.trial_info["target_angle"] = rel_tgt_angle
        # binning the target angles into num_conds bins
        _, bins = np.histogram(ds.trial_info["target_angle"].dropna(), bins=8, range=(0.0,360.0))
        categories = ds.trial_info["target_angle"].dropna().apply(lambda x: np.digitize(x, bins, right=False))
        ds.trial_info["target_condition"] = categories

        # computing movement onset time
        speed = np.sqrt(ds.data['cursor_vel'].x ** 2 + ds.data['cursor_vel'].y **2)
        ds.data['speed'] = speed
        ds.get_backward_move_onset(move_field = 'speed',
                                    start_field="start_time",
                                    win_start_offset=100,
                                    threshold = 0.3,
                                    ignored_trials=None)
        
        ignored_trials = ((ds.trial_info['result'] != 'S') | 
                            (ds.trial_info['different_move_onset_time'] == True) | 
                            (ds.trial_info['has_pause'] == True) |
                            (ds.trial_info['has_timing_issues'] == True) |
                            (ds.trial_info['trial_type'] != 'L') |
                            ds.trial_info['forward_move_onset_time'].isnull())
        
        ds.trials = ds.make_trial_data(
            align_field='forward_move_onset_time',
            align_range=[-250,500],  # ms
            allow_overlap=False,
            ignored_trials=ignored_trials,
            allow_nans=True)
        
        # condition-average the trials
        psth = PSTH(ds.trial_info["target_condition"])
        spikes_mean, _ = psth.compute_trial_average(ds.trials, 'spikes_smooth',ignore_nans=True)
        rates_mean, _ = psth.compute_trial_average(ds.trials, 'lfads_rates',ignore_nans=True)
        spikes_means_pivot = spikes_mean.pivot(index='align_time', columns='condition_id')
        spikes_mean_data = np.stack(
            np.split(spikes_means_pivot.values, len(np.unique(ds.data.spikes.columns)), axis=1)
        )
        rates_means_pivot = rates_mean.pivot(index='align_time', columns='condition_id')
        rates_mean_data = np.stack(
            np.split(rates_means_pivot.values, len(np.unique(ds.data.spikes.columns)), axis=1)
        )
        heldout_channels = heldout_idx
        heldin_channels = heldin_idx
        # heldin_channels = [i for i in range(spikes_mean_data.shape[0]) if i not in heldout_channels]
        R2_heldin = np.full((len(heldin_channels),), np.nan)
        R2_heldout = np.full((len(heldout_channels),), np.nan)
        # compute the R2
        for i in range(len(heldin_channels)):
            R2_heldin[i] = sklearn.metrics.r2_score(spikes_mean_data[heldin_channels[i],:,:].swapaxes(0,1).flatten(), rates_mean_data[heldin_channels[i],:,:].swapaxes(0,1).flatten())
        for i in range(len(heldout_channels)):
            R2_heldout[i] = sklearn.metrics.r2_score(spikes_mean_data[heldout_channels[i],:,:].swapaxes(0,1).flatten(), rates_mean_data[heldout_channels[i],:,:].swapaxes(0,1).flatten())

        SESSION = ds.fpath.split('/')[-1].split('.')[0]
        # get the SNR array
        with open(f'/snel/share/data/neuralink/snr/pager/snr_{SESSION}.pkl', 'rb') as f:
            snr_array = pickle.load(f)
        snr_array['snr'] = snr_array['snr'][channels.astype(np.int32)]
        snr_heldin = snr_array['snr'][heldin_channels]
        snr_heldout = snr_array['snr'][heldout_channels]

        psth_r2_heldin = R2_heldin[np.where(snr_heldin > 0)[0]].mean()
        psth_r2_heldout = R2_heldout[np.where(snr_heldout > 0)[0]].mean()
        
        num_trials = len(np.unique(ds.trials.trial_id.values))
        trials_with_nans = np.unique(ds.trials.trial_id.loc[np.isnan(ds.trials['joystick_position'].values)].values)
        if len(trials_with_nans)/num_trials > 0.33: 
            print(f'Skipping this dataset - {len(trials_with_nans)} contain NaNs.')
            behavior_r2_heldin = np.nan
            behavior_r2_heldout = np.nan
        else:
            print(f'Removing {len(trials_with_nans)} trials that contain NaNs.')
            ds.trials = ds.trials.loc[np.isin(ds.trials.trial_id.values, trials_with_nans, invert=True)]
            
            (x_train, y_train, train_ids), (x_valid, y_valid, valid_ids) = \
                    dec.prepare_decoding_data(
                        ds.trials,
                        'lfads_factors',
                        'joystick_position',
                        valid_ratio=0.2,
                        ms_lag=0,
                        n_history=0,
                        # channels = heldin_channels,
                        return_groups=True,
                    )

            decoder = dec.NeuralDecoder(        
                    {
                        'estimator': sklearn.linear_model.Ridge(), 
                        'param_grid': {'alpha': np.logspace(2, 3, 10)},
                        'cv': 5,
                    }
                )

            decoder.fit(x_train, y_train)
            behavior_r2_heldin = decoder.score(x_valid, y_valid, multioutput='variance_weighted')

            (x_train, y_train, train_ids), (x_valid, y_valid, valid_ids) = \
                    dec.prepare_decoding_data(
                        ds.trials,
                        'lfads_factors',
                        'joystick_position',
                        valid_ratio=0.2,
                        ms_lag=0,
                        n_history=0,
                        # channels = heldout_channels,
                        return_groups=True,
                    )

            decoder = dec.NeuralDecoder(        
                    {
                        'estimator': sklearn.linear_model.Ridge(), 
                        'param_grid': {'alpha': np.logspace(2, 3, 10)},
                        'cv': 5,
                    }
                )

            decoder.fit(x_train, y_train)
            behavior_r2_heldout = decoder.score(x_valid, y_valid, multioutput='variance_weighted')

        # Co-BPS calculation
        bps = bits_per_spike(valid_heldout_pred, valid_heldout_truth)

        # writer.log_metrics({'post_behavior_r2': r2, 'post_co_bps': bps})
      
        metrics = {
            'kl_ic_scale': model.hparams.kl_ic_scale,
            'kl_co_scale': model.hparams.kl_co_scale,
            'l2_con_scale': model.hparams.l2_con_scale,
            'l2_gen_scale': model.hparams.l2_gen_scale,
            'dropout_rate': model.hparams.dropout_rate,
            # 'cd_rate': model.hparams.train_aug_stack.batch_transforms[1].cd_rate,
            'lr_init': model.hparams.lr_init,
            'batch_size': dhps.batch_size,
            'ic_enc_dim': model.hparams.ic_enc_dim,
            'ci_enc_dim': model.hparams.ci_enc_dim,
            'con_dim': model.hparams.con_dim,
            'co_dim': model.hparams.co_dim,
            'ic_dim': model.hparams.ic_dim,
            'gen_dim': model.hparams.gen_dim,
            'fac_dim': model.hparams.fac_dim,
            'heldin_channels': len(heldin_channels),
            'heldout_channels': len(heldout_channels),
            'behavior_r2_heldin': behavior_r2_heldin,
            'behavior_r2_heldout': behavior_r2_heldout,
            'post_co_bps': bps,
            'psth_r2_heldin': psth_r2_heldin,
            'psth_r2_heldout': psth_r2_heldout,
        }
        json_object = json.dumps(metrics)
 
        # Writing to sample.json
        with open("evaluation_metrics.json", "w") as outfile:
            outfile.write(json_object)
        