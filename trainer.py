from typing import List, Union
import torch
import pytorch_lightning as pl
import numpy as np
from lit_modules import TD3LitModule, TD3LitDataModule
from master_config import MasterConfig



class BaseTrainer:
    """
    BaseTrainer:
        Base class which all other TD3 trainers derive from.

    config (MasterConfig):  Master configuration instance.
    """

    def __init__(self, config: MasterConfig) -> None:
        self.config = config
        self._setup()

    def run(self) -> None:
        # Overwrite
        raise NotImplementedError()

    def _setup(self) -> None:
        self._data_module = self._get_data_module()
        self._lit_module = self._get_lit_module()
        self._callbacks = self._get_callbacks()
        self._loggers = self._get_loggers()
        self._strategy = self._get_strategy()
        self._lit_trainer = self._get_lit_trainer()
        torch.manual_seed(self.config.RANDOM_SEED)
        self._data_module.env.action_space.seed(self.config.RANDOM_SEED)
        np.random.seed(self.config.RANDOM_SEED)

    def _get_data_module(self) -> pl.LightningDataModule:
        # Overwrite
        raise NotImplementedError()

    def _get_lit_module(self) -> pl.LightningModule:
        # Overwrite
        raise NotImplementedError()

    def _get_callbacks(self) -> List[pl.callbacks.Callback]:
        # Overwrite
        raise NotImplementedError()

    def _get_loggers(self) -> List[pl.loggers.base.LightningLoggerBase]:
        # Overwrite
        raise NotImplementedError()

    def _get_strategy(self) -> Union[None, any]:
        # Overwrite
        raise NotImplementedError()

    def _get_lit_trainer(self) -> pl.Trainer:
        # Overwrite
        raise NotImplementedError()


class StandardTD3Trainer(BaseTrainer):

    def __init__(self, config: MasterConfig) -> None:
        super().__init__(config)

    def run(self) -> None:
        self._lit_trainer.fit(self._lit_module,
                              self._data_module)

    def _get_data_module(self) -> pl.LightningDataModule:
        data_module = TD3LitDataModule(self.config)
        data_module.setup()
        self.config.ACTION_SPACE_LEN = \
            data_module.env.action_space.shape[0]
        self.config.STATE_DIMS = \
            len(data_module.env.observation_space.shape)
        self.config.STATE_SPACE_LEN = \
            data_module.env.observation_space.shape[0]
        return data_module

    def _get_lit_module(self) -> pl.LightningModule:
        Lit_module = TD3LitModule(self.config)
        return Lit_module

    def _get_callbacks(self) -> List[pl.callbacks.Callback]:
        callbacks = []
        # checkpointing
        callbacks.append(pl.callbacks.ModelCheckpoint(
                            monitor='Average Return', dirpath=('./experiments/'
                            f'{self.config.ENVIRONMENT_API}/'
                            f'{self.config.EXPERIMENT_NAME}'),
                            filename='{epoch:02d}-{reward:.4f}',
                            save_top_k=self.config.SAVE_TOP_K,
                            mode='max', save_last=True,
                            every_n_train_steps=self.config.VAL_CHECK_INTERVAL))
        # Fancy progress bar
        callbacks.append(CustomProgressBar())
        return callbacks

    def _get_loggers(self) -> List[pl.loggers.base.LightningLoggerBase]:
        # Setup loggers
        loggers = []
        loggers.append(pl.loggers.TensorBoardLogger('./experiments/',
                    name=f'{self.config.ENVIRONMENT_API}', 
                    version=self.config.EXPERIMENT_NAME))
        return loggers

    def _get_strategy(self) -> Union[None, any]:
        return None

    def _get_lit_trainer(self) -> pl.Trainer:
        # Setup trainer
        trainer = pl.Trainer(
            gpus=self.config.GPUS,
            callbacks=self._callbacks, 
            logger=self._loggers,
            val_check_interval=0,
            num_sanity_val_steps=0,
            max_steps=self.config.MAX_STEPS)
        return trainer


class CustomProgressBar(pl.callbacks.progress.ProgressBar):

    def on_train_epoch_start(self, trainer, pl_module):
        super().on_train_epoch_start(trainer, pl_module)
        self.main_progress_bar.set_description(f"Step {pl_module.n_iterations}")

if __name__ == '__main__':
    trainer = StandardTD3Trainer(MasterConfig)
    trainer.run()