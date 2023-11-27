import multiprocessing
import os
import shutil
from glob import glob

import pandas as pd
import torch

from .tuples import SessionBatch


def flatten(dictionary, level=[]):
    """Flattens a dictionary by placing '.' between levels.
    This function flattens a hierarchical dictionary by placing '.'
    between keys at various levels to create a single key for each
    value. It is used internally for converting the configuration
    dictionary to more convenient formats. Implementation was
    inspired by `this StackOverflow post
    <https://stackoverflow.com/questions/6037503/python-unflatten-dict>`_.
    Parameters
    ----------
    dictionary : dict
        The hierarchical dictionary to be flattened.
    level : str, optional
        The string to append to the beginning of this dictionary,
        enabling recursive calls. By default, an empty string.
    Returns
    -------
    dict
        The flattened dictionary.
    """

    tmp_dict = {}
    for key, val in dictionary.items():
        if type(val) == dict:
            tmp_dict.update(flatten(val, level + [key]))
        else:
            tmp_dict[".".join(level + [key])] = val
    return tmp_dict


def transpose_lists(output: list[list]):
    """Transposes the ordering of a list of lists."""
    return list(map(list, zip(*output)))


def send_batch_to_device(batch, device):
    """Recursively searches the batch for tensors and sends them to the device"""

    def send_to_device(obj):
        obj_type = type(obj)
        if obj_type == torch.Tensor:
            return obj.to(device)
        elif obj_type == dict:
            return {k: send_to_device(v) for k, v in obj.items()}
        elif obj_type == list:
            return [send_to_device(o) for o in obj]
        elif obj_type == SessionBatch:
            return SessionBatch(*[send_to_device(o) for o in obj])
        else:
            raise NotImplementedError(
                f"`send_batch_to_device` has not been implemented for {str(obj_type)}."
            )

    return send_to_device(batch)


def read_pbt_fitlog(pbt_dir):
    """Compiles fitlogs of all PBT workers in a directory into a single DataFrame"""
    worker_logs = sorted(glob(pbt_dir + "/run_model_*/csv_logs/version_*/metrics.csv"))
    with multiprocessing.Pool(8) as p:
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


def cleanup_best_model(model_dir):
    """Deletes extra files in the best_model folder after PBT"""
    # Find the most recently created checkpoint file and move it
    ckpt_files = glob(model_dir + "/checkpoint_epoch=*/tune.ckpt")
    ckpt_file = sorted(ckpt_files, key=os.path.getctime)[-1]
    shutil.copyfile(ckpt_file, model_dir + "/model.ckpt")
    # Delete irrelevant / confusing files
    cleanup_files = glob(model_dir + "/events.out.tfevents*")
    cleanup_files.extend(glob(model_dir + "/checkpoint_*"))
    cleanup_files.extend(
        [
            model_dir + "/csv_logs",
            model_dir + "/hparams.yaml",
            model_dir + "/params.json",
            model_dir + "/params.pkl",
            model_dir + "/progress.csv",
            model_dir + "/result.json",
        ]
    )
    for f in cleanup_files:
        if os.path.isdir(f):
            shutil.rmtree(f)
        else:
            os.remove(f)
