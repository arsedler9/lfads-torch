import pytorch_lightning as pl

# This strategy class prevents PTL from restoring optimizers during PBT


class SingleDeviceStrategy(pl.strategies.SingleDeviceStrategy):
    @property
    def lightning_restore_optimizer(self) -> bool:
        """Override to disable Lightning restoring optimizers/schedulers.
        This is useful for plugins which manage restoring optimizers/schedulers.
        """
        return False
