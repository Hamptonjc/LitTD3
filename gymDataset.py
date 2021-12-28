import collections
import numpy as np
from typing import Tuple, NamedTuple
import torch
import pytorch_lightning as pl
import gym
from agent import Agent


#Experience = collections.namedtuple(
#    'Experience', field_names=['state', 'action', 'reward',
#                               'done', 'new_state'])

# Named tuple for storing experience steps gathered in training
class Experience(NamedTuple):
    state: np.array
    action: float
    reward: float
    done: bool
    next_state: np.array


# from https://www.pytorchlightning.ai/blog/en-lightning-reinforcement-learning-building-a-dqn-with-pytorch-lightning
class ReplayBuffer:
    """
    Replay Buffer for storing past experiences allowing the agent to learn from them

    Args:
        capacity: size of the buffer
    """

    def __init__(self, capacity: int) -> None:
        self.buffer = collections.deque(maxlen=capacity)

    def __len__(self) -> None:
        return len(self.buffer)

    def append(self, experience: Experience) -> None:
        """
        Add experience to the buffer

        Args:
            experience: tuple (state, action, reward, done, new_state)
        """
        self.buffer.append(experience)

    def sample(self, batch_size: int) -> Tuple:
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        states, actions, rewards, dones, next_states = zip(*[self.buffer[idx] for idx in indices])

        return (np.array(states), np.array(actions), np.array(rewards, dtype=np.float32),
                np.array(dones, dtype=np.bool), np.array(next_states))


class GymDataset(torch.utils.data.IterableDataset):
    """
    Iterable Dataset containing the ReplayBuffer
    which will be updated with new experiences during training
    Args:
        buffer: replay buffer
        sample_size: number of experiences to sample at a time
    """
    def __init__(self, buffer: ReplayBuffer, sample_size: int = 200) -> None:
        self.buffer = buffer
        self.sample_size = sample_size

    def __iter__(self) -> Tuple:
        states, actions, rewards, dones, new_states = self.buffer.sample(self.sample_size)
        for i in range(len(dones)):
            yield states[i], actions[i], rewards[i], dones[i], new_states[i]


class LitGymDataset(pl.LightningDataModule):

    def __init__(self, config: object) -> None:
        self.config = config
        self.replay_buffer = ReplayBuffer(config.BUFFER_SIZE)

    def setup(self, stage: str=None)->None:
        # Environment
        self.env = gym.make(self.config.GYM_ENVIRONMENT)
        # Agent
        self.agent = Agent(self.env, self.replaybuffer)

    def train_dataloader(self) -> torch.utils.data.DataLoader:
        dataset = GymDataset(self.replay_buffer, self.config.SAMPLE_SIZE)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=self.config.BATCH_SIZE)
        return dataloader

    def val_dataloader(self) -> torch.utils.data.DataLoader:
        dataset = GymDataset(self.replay_buffer, self.config.SAMPLE_SIZE)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=self.config.BATCH_SIZE)
        return dataloader
