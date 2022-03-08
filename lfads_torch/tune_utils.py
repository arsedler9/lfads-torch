def trial_dirname_creator(trial):
    """Uses only the trial name to create the directory, as
    opposed to the default behavior of using sampled HPs.
    """
    return str(trial)
