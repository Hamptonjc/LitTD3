from typing import Iterator
import gym
import numpy as np
import torch, torchvision
import pytorch_lightning as pl
from agent import ReplayBuffer


class TD3DatasetBase(torch.utils.data.IterableDataset):
    """
    TD3DatasetBase: Base class which all other TD3 datasets are derived from.
        This is an iterable Dataset containing the ReplayBuffer
        which will be updated with new experiences during training
    Args:
        buffer (ReplayBuffer): buffer object containing the recorded experiences.
        batch_size (int): number of experiences to sample at a time.
    """
    def __init__(self, buffer: ReplayBuffer, batch_size: int) -> None:
        self.buffer = buffer
        self.batch_size = batch_size

    def __iter__(self) -> Iterator:
        (states, actions, rewards,
         dones, new_states) = self.buffer.sample(self.batch_size)
        for i in range(len(dones)):
            (state, action, reward,
             done, new_state) = self.preprocess((states[i], actions[i],
                                     rewards[i], dones[i], new_states[i]))
            yield state, action, reward, done, new_state 

    def preprocess(self, experience):
        state, action, reward, done, new_state = experience
        if len(state.shape) > 1:
            state = torchvision.transforms.ToTensor()(
                                state.astype(np.uint8))
            new_state = torchvision.transforms.ToTensor()(
                                new_state.astype(np.uint8))
        else:
            state = torch.tensor(state, dtype=torch.float32)
            new_state = torch.tensor(new_state, dtype=torch.float32)
        done = torch.tensor([done], dtype=torch.int)
        action = torch.tensor(action, dtype=torch.float32)
        reward = torch.tensor([reward], dtype=torch.float32)
        return state, action, reward, done, new_state


class TwoDimStateGymDataset(TD3DatasetBase):
    """
    TwoDimStateGymDataset: Example dataset for a 
                    2-dimensional state space task.
    Args:
        buffer (ReplayBuffer): buffer object containing 
                                the recorded experiences.
        batch_size (int): number of experiences to sample at a time.
    """
    def __init__(self, buffer: ReplayBuffer, batch_size: int) -> None:
        super().__init__(buffer, batch_size)

    def preprocess(self, experience):
        state, action, reward, done, new_state = experience
        state = torchvision.transforms.ToTensor()(state.astype(np.uint8))
        new_state = torchvision.transforms.ToTensor()(new_state.astype(np.uint8))
        done = torch.tensor([done], dtype=torch.int)
        action = torch.tensor(action, dtype=torch.float32)
        reward = torch.tensor([reward], dtype=torch.float32)
        return state, action, reward, done, new_state


class OneDimStateGymDataset(TD3DatasetBase):
    """
    OneDimStateGymDataset: Example dataset for a 
                    1-dimensional state space task.
    Args:
        buffer (ReplayBuffer): buffer object containing 
                                the recorded experiences.
        batch_size (int): number of experiences to sample at a time.
    """
    def __init__(self, buffer: ReplayBuffer, batch_size: int) -> None:
        super().__init__(buffer, batch_size)

    def preprocess(self, experience):
        state, action, reward, done, new_state = experience
        state = torch.tensor(state, dtype=torch.float32)
        new_state = torch.tensor(new_state, dtype=torch.float32)
        done = torch.tensor([done], dtype=torch.int)
        action = torch.tensor(action, dtype=torch.float32)
        reward = torch.tensor([reward], dtype=torch.float32)
        return state, action, reward, done, new_state
