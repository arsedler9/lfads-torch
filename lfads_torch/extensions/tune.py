import copy
import logging
import multiprocessing
import random
from glob import glob
from typing import Callable, Dict, List, Optional, Tuple, Union
from collections import deque

import numpy as np
import pandas as pd
from ray.air._internal.checkpoint_manager import CheckpointStorage
from ray.tune.execution import trial_runner
from ray.tune.experiment import Trial
from ray.tune.schedulers import PopulationBasedTraining
from ray.tune.search.sample import Domain
from ray.tune.stopper import Stopper

logger = logging.getLogger(__name__)


def read_pbt_fitlog(pbt_dir, n_processes=8):
    """Compiles fitlogs of all PBT workers in a directory into a single DataFrame"""
    worker_logs = sorted(glob(pbt_dir + "/run_model_*/csv_logs/version_*/metrics.csv"))
    with multiprocessing.Pool(n_processes) as p:
        fit_dfs = p.map(pd.read_csv, worker_logs)
    for i, df in enumerate(fit_dfs):
        df = (
            df[~df.epoch.isnull()]
            .dropna(axis=1, how="all")
            .ffill()
            .drop_duplicates(subset="epoch", keep="last")
        )
        df["worker_id"] = i
        fit_dfs[i] = df
    fit_df = pd.concat(fit_dfs).reset_index(drop=True)
    return fit_df


def _explore(
    config: Dict,
    mutations: Dict,
) -> Tuple[Dict, Dict]:
    """Return a perturbed config and string descriptors of the operations performed
    on the original config to produce the new config.

    Args:
        config: Original hyperparameter configuration.
        mutations: Specification of mutations to perform as documented
            in the PopulationBasedTraining scheduler.

    Returns:
        new_config: New hyperparameter configuration (after random mutations).
        operations: Map of hyperparams -> strings describing mutation operations
            performed
    """
    operations = {}
    new_config = copy.deepcopy(config)
    for name, hp in mutations.items():
        # resample until new_value is within the bounds
        new_value = -np.inf
        # keep track of the number perturbation attempts
        num_tries = 0
        while not hp.min_bound <= new_value <= hp.max_bound:
            # sample in the linear space
            min_perturb = 0.002 * hp.explore_wt
            max_perturb = hp.explore_wt
            # take a sample from the scaling space
            perturbation = random.choice(
                [
                    random.uniform(min_perturb, max_perturb),
                    random.uniform(-max_perturb, -min_perturb),
                ]
            )
            # center the perturbation at 1
            scale = 1 + perturbation
            # compute the new value candidate
            new_value = scale * new_config[name]
            # if something goes wrong with perturbation, clip
            num_tries += 1
            if num_tries > 99:
                # clip this value to the boundaries
                new_value = np.clip(new_value, hp.min_bound, hp.max_bound)
        # assign the final value to the new config
        new_config[name] = float(new_value)
        operations[name] = f"* {scale}"

    return new_config, operations


class HyperParam:
    def __init__(
        self,
        min_samp: float,
        max_samp: float,
        init: Union[float, Callable] = None,
        sample_fn: str = "loguniform",
        explore_wt: float = 0.2,
        enforce_limits: bool = False,
    ):
        """Represents constraints on hyperparameter
        values that will be used during PBT.

        Parameters
        ----------
        min_samp : float
            The minimum allowed sample
        max_samp : float
            The maximum allowed sample
        init : float
            The initial value to use for PBT, by default
            None initializes with a sample from the distribution.
        sample_fn : {'loguniform', 'uniform', 'randint'} or callable, optional
            The distribution from which to sample, by default
            'loguniform'
        explore_wt : float, optional
            The maximum percentage increase or decrease for
            a perturbation, by default 0.2
        enforce_limits : bool, optional
            Whether to limit exploration to within the sample_range,
            by default False

        Raises
        ------
        ValueError
            When an invalid sample_fn is provided.
        """

        # check the sampling range
        assert min_samp < max_samp, "`min_samp` must be smaller than `max_samp`."
        # set up the sampling function
        if callable(sample_fn):
            self.sample = sample_fn
        elif sample_fn in ["uniform", "loguniform", "randint"]:
            if sample_fn == "loguniform":
                base = 10
                logmin = np.log(min_samp) / np.log(base)
                logmax = np.log(max_samp) / np.log(base)
                self.sample = lambda _: base ** (np.random.uniform(logmin, logmax))
            else:
                self.sample = lambda _: getattr(np.random, sample_fn)(
                    low=min_samp, high=max_samp
                )
        else:
            raise ValueError("Invalid `sample_fn` was specified.")
        # use the initial value if specified, otherwise use sampling function
        self.init = (lambda _: init) if init is not None else self.sample
        # save other attributes
        self.min_bound = min_samp if enforce_limits else 0
        self.max_bound = max_samp if enforce_limits else np.inf
        self.explore_wt = explore_wt

    def __call__(self):
        return self.sample()


class BinaryTournamentPBT(PopulationBasedTraining):
    def __init__(
        self,
        time_attr: str = "training_iteration",
        metric: Optional[str] = None,
        mode: Optional[str] = None,
        perturbation_interval: float = 60.0,
        burn_in_period: float = 0.0,
        hyperparam_mutations: Dict[
            str, Union[dict, list, tuple, Callable, Domain]
        ] = None,
    ):
        super().__init__(
            time_attr=time_attr,
            metric=metric,
            mode=mode,
            perturbation_interval=perturbation_interval,
            burn_in_period=burn_in_period,
            hyperparam_mutations=hyperparam_mutations,
            quantile_fraction=0.0,
            resample_probability=0.0,
            perturbation_factors=None,
            custom_explore_fn=None,
            log_config=True,
            require_attrs=True,
            synch=True,
        )

    def _checkpoint_or_exploit(
        self,
        trial: Trial,
        trial_runner: "trial_runner.TrialRunner",
        upper_quantile: List[Trial],
        lower_quantile: List[Trial],
    ):
        """Checkpoint if in upper quantile, exploits if in lower."""
        trial_executor = trial_runner.trial_executor
        state = self._trial_state[trial]
        if trial in upper_quantile:
            # The trial last result is only updated after the scheduler
            # callback. So, we override with the current result.
            logger.debug("Trial {} is in upper quantile".format(trial))
            logger.debug("Checkpointing {}".format(trial))
            if trial.status == Trial.PAUSED:
                # Paused trial will always have an in-memory checkpoint.
                state.last_checkpoint = trial.checkpoint
            else:
                state.last_checkpoint = trial_executor.save(
                    trial, CheckpointStorage.MEMORY, result=state.last_result
                )
            self._num_checkpoints += 1
        else:
            state.last_checkpoint = None  # not a top trial

        if trial in lower_quantile:
            logger.debug("Trial {} is in lower quantile".format(trial))
            trial_to_clone = upper_quantile[lower_quantile.index(trial)]
            assert trial is not trial_to_clone
            if not self._trial_state[trial_to_clone].last_checkpoint:
                logger.info(
                    "[pbt]: no checkpoint for trial."
                    " Skip exploit for Trial {}".format(trial)
                )
                return
            self._exploit(trial_runner, trial, trial_to_clone)

    def _get_new_config(self, trial: Trial, trial_to_clone: Trial) -> Tuple[Dict, Dict]:
        """Gets new config for trial by exploring trial_to_clone's config.

        Args:
            trial: The current trial that decided to exploit trial_to_clone.
            trial_to_clone: The top-performing trial with a hyperparameter config
                that the current trial will explore by perturbing.

        Returns:
            new_config: New hyperparameter configuration (after random mutations).
            operations: Map of hyperparams -> strings describing mutation operations
                performed
        """
        return _explore(
            trial_to_clone.config,
            self._hyperparam_mutations,
        )

    def _quantiles(self) -> Tuple[List[Trial], List[Trial]]:
        """Returns trials in the lower and upper `quantile` of the population.

        If there is not enough data to compute this, returns empty lists.
        """
        trials = []
        for trial, state in self._trial_state.items():
            logger.debug("Trial {}, state {}".format(trial, state))
            if trial.is_finished():
                logger.debug("Trial {} is finished".format(trial))
            if state.last_score is not None and not trial.is_finished():
                trials.append(trial)

        if len(trials) <= 1:
            return [], []
        else:
            # ----- Binary Tournament -----
            random.shuffle(trials)
            num_competitions = len(trials) // 2
            c1, c2 = trials[:num_competitions], trials[-num_competitions:]
            winners, losers = [], []
            for t1, t2 in zip(c1, c2):
                if self._trial_state[t1].last_score > self._trial_state[t2].last_score:
                    winners.append(t1)
                    losers.append(t2)
                else:
                    winners.append(t2)
                    losers.append(t1)
            return losers, winners

class PercentageChangeStopper(Stopper):
    """Stops the experiment early when the best score has not improved by a specified amount within
    a specified number of intermediate results.
    """
    def __init__(
        self,
        metric: str = "valid/recon_smth",
        patience: int = 4,
        min_percent_improvement: float = .0005,
        ramp_up: int = 80,
        pert_int: int = 25,
        num_trials: int = 16
        #params
    ):
        #TODO catch errors for params
        
        self._metric = metric
        self._patience = patience
        self._min_percent_improvement = min_percent_improvement
        self._percent_improvement = 0.0
        self._best_scores = deque(maxlen=patience)
        self._ramp_up = ramp_up
        self._current_gen = 0
        self._pert_int = pert_int
        self._scores_by_generation = {}
        self._num_trials = num_trials
        self._last_completed_gen = -1

    def __call__(self, trial_id, result):
        effectiveGen = np.floor((result["training_iteration"]-1-self._ramp_up)/self._pert_int)

        if effectiveGen >= 0: #is done ramping up
            if effectiveGen in self._scores_by_generation: #not first in generation
                self._scores_by_generation[effectiveGen].append(result[self._metric])
            else: #first in generation
                self._scores_by_generation[effectiveGen] = [result[self._metric]]
           
            if len(self._scores_by_generation[self._current_gen]) == self._num_trials*self._pert_int:
                
                best_score = np.min(self._scores_by_generation[self._current_gen])
                self._last_completed_gen = self._current_gen
                self._current_gen += 1
                #check/update conditions, ask stop_all if we stop
                self._best_scores.append(best_score)
                self._percent_improvement = (
                    self._best_scores[0] - np.min(self._best_scores)
                ) / np.mean(np.abs(self._best_scores))
        return self.stop_all()

    def failed_to_improve(self):
        return self._percent_improvement <= self._min_percent_improvement
        
    def stop_all(self):
        return self.failed_to_improve() and self._last_completed_gen >= self._patience