from typing import Iterator
import torch
import pytorch_lightning as pl
import gym
from agent import *


#Experience = collections.namedtuple(
#    'Experience', field_names=['state', 'action', 'reward',
#                               'done', 'new_state'])

class GymDataset(torch.utils.data.IterableDataset):
    """
    Iterable Dataset containing the ReplayBuffer
    which will be updated with new experiences during training
    Args:
        buffer: replay buffer
        sample_size: number of experiences to sample at a time
    """
    def __init__(self, buffer: ReplayBuffer, sample_size: int) -> None:
        self.buffer = buffer
        self.sample_size = sample_size

    def __iter__(self) -> Iterator:
        states, actions, rewards, dones, new_states = self.preprocess(self.buffer.sample(self.sample_size))
        for i in range(len(dones)):
            yield states[i], actions[i], rewards[i], dones[i], new_states[i]

    def preprocess(self, experiences):
        states, actions, rewards, dones, new_states = experiences
        states = states.transpose(0,3,1,2)
        rewards = np.expand_dims(rewards, 1)
        dones = np.expand_dims(dones, 1)
        new_states = new_states.transpose(0,3,1,2)
        return states, actions, rewards, dones, new_states



class LitGymDataset(pl.LightningDataModule):

    def __init__(self, config: object) -> None:
        self.config = config
        self.replay_buffer = ReplayBuffer(config.BUFFER_SIZE)

    def setup(self, stage: str=None)->None:
        # Environment
        self.env = gym.make(self.config.GYM_ENVIRONMENT)
        # Agent
        self.agent = Agent(self.env, self.replay_buffer)

    def train_dataloader(self) -> torch.utils.data.DataLoader:
        dataset = GymDataset(self.replay_buffer, self.config.EPISODE_LENGTH)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=self.config.BATCH_SIZE)
        return dataloader

#    def val_dataloader(self) -> torch.utils.data.DataLoader:
#        dataset = GymDataset(self.replay_buffer, self.config.SAMPLE_SIZE)
#        dataloader = torch.utils.data.DataLoader(dataset, batch_size=self.config.BATCH_SIZE)
#        return dataloader
