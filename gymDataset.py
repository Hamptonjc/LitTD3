from typing import Iterator
import torch, torchvision
import pytorch_lightning as pl
import gym
from agent import *


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
        states, actions, rewards, dones, new_states = self.buffer.sample(self.sample_size)
        for i in range(len(dones)):
            state, action, reward, done, new_state = self.preprocess((states[i], actions[i], rewards[i], dones[i], new_states[i]))
            yield state, action, reward, done, new_state 

    def preprocess(self, experience):
        state, action, reward, done, new_state = experience
        state = torchvision.transforms.ToTensor()(state.astype(np.uint8))
        new_state = torchvision.transforms.ToTensor()(new_state.astype(np.uint8))
        return state, action, reward, done, new_state

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
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=self.config.BATCH_SIZE, num_workers=8)
        return dataloader

#    def val_dataloader(self) -> torch.utils.data.DataLoader:
#        dataset = GymDataset(self.replay_buffer, self.config.SAMPLE_SIZE)
#        dataloader = torch.utils.data.DataLoader(dataset, batch_size=self.config.BATCH_SIZE)
#        return dataloader
