import torch
import pytorch_lightning as pl
from litTD3 import LitTD3
from gymDataset import LitGymDataset
from config import TD3Config as config


def train() -> None:
    # Seed RNG
    torch.manual_seed(314)
    # Instatiate lit Data
    lit_gym_ds = LitGymDataset(config)
    lit_gym_ds.setup()
    # Get environment info
    n_actions = lit_gym_ds.env.action_space.shape[0]
    state_dims = len(lit_gym_ds.env.observation_space.shape)
    if state_dims == 1:
        state_len = lit_gym_ds.env.observation_space.shape[0]
    else:
        state_len = None
    # Instatiate litTD3
    lit_model = LitTD3(config, n_actions, state_dims, state_len)
    # Add callbacks
    callbacks = []
    # checkpointing
    callbacks.append(pl.callbacks.ModelCheckpoint(monitor='reward',
                                                  dirpath='./experiments/' + \
                                                       f'{config.GYM_ENVIRONMENT}-TD3/' + \
                                                        config.EXPERIMENT_NAME,
                                                  filename='{epoch:02d}-{reward:.10f}',
                                                  save_top_k=config.SAVE_TOP_K,
                                                  mode='max',
                                                  save_last=True,
                                                  every_n_train_steps=config.VAL_CHECK_INTERVAL
                                                  ))
    # Fancy progress bar
    callbacks.append(pl.callbacks.RichProgressBar())
    
    # Setup logger
    logger = pl.loggers.TensorBoardLogger('./experiments/',
                                          name=f'{config.GYM_ENVIRONMENT}-TD3',
                                          version=config.EXPERIMENT_NAME)
    # Setup trainer
    trainer = pl.Trainer(gpus=config.GPUS,
                         callbacks=callbacks,
                         logger=logger,
                         val_check_interval=config.VAL_CHECK_INTERVAL,
                         num_sanity_val_steps=0,
                         max_steps=config.MAX_STEPS)

    # Fit
    trainer.fit(model=lit_model, datamodule=lit_gym_ds)


if __name__ == '__main__':
    train()