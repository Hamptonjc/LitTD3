import torch
import pytorch_lightning as pl
from .litTD3 import LitTD3
from .gymDataset import LitGymDataset
from .config import TD3Config as config


def train():
    # Seed RNG
    torch.manual_seed(314)
    # Instatiate lit Data
    lit_gym_ds = LitGymDataset(config)
    # Instatiate litTD3
    lit_model = LitTD3(config)
    # Add callbacks
    callbacks = []
    # checkpointing
    callbacks.append(pl.callbacks.ModelCheckpoint(
                     monitor='reward',
                     dirpath=config.SAVED_MODEL_DIR,
                     filename='{epoch:02d}-{val_loss:.10f}',
                     save_top_k=config.SAVE_TOP_K,
                     mode='max',
                     save_last=True
                    ))
    # Setup logger
    logger = pl.loggers.TensorBoardLogger(config.LOG_DIR,
                                          name=f'{config.GYM_ENVIRONMENT}-TD3',
                                          version=config.RUN_NAME)
    # Setup trainer
    trainer = pl.Trainer(gpus=config.GPUS,
                         callbacks=callbacks,
                         logger=logger,
                         val_check_interval=100)
    # Fit
    trainer.fit(model=lit_model, datamodule=lit_gym_ds)


if __name__ == '__main__':
    train()