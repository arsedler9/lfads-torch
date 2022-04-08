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


def batch_fwd(model, batch, sample_posteriors=False):
    """Performs the forward pass for a given data batch.

    Parameters
    ----------
    model : lfads_torch.models.base_model.LFADS
        The model to pass data through.
    batch : tuple[torch.Tensor]
        A tuple of batched input tensors.

    Returns
    -------
    tuple[torch.Tensor]
        A tuple of batched output tensors.
    """
    input_data, ext = batch[0], batch[3]
    return model(
        input_data.to(model.device),
        ext.to(model.device),
        sample_posteriors=sample_posteriors,
    )


def get_batch_fwd():
    """Utility function for accessing the `batch_fwd` function
    from `hydra` configs.
    """
    return batch_fwd
