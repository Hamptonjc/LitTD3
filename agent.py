# Imports
import collections
from typing import Tuple, NamedTuple
import torch
import gym
import numpy as np


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

        return (np.array(states, dtype=np.float32), np.array(actions, dtype=np.float32), np.array(rewards, dtype=np.float32),
                np.array(dones, dtype=np.bool), np.array(next_states, dtype=np.float32))


class Agent:

    def __init__(self, env: gym.Env, replay_buffer: ReplayBuffer) -> None:
        self.env = env
        self.replay_buffer = replay_buffer
        self.state = self.env.reset()

    def reset(self) -> None:
        self.state = self.env.reset()

    def get_action(self, policy_network: torch.nn.Module, noise: float) -> np.array:
        # Convert state from numpy array to torch tensor
        state = torch.tensor((self.state.copy()).astype(np.float32))
        state = state.permute(2,1,0)
        state = state.unsqueeze(0)
        # get action from policy network
        action = policy_network(state)
        if noise != 0.0:
            noise = torch.normal(0, noise, size=action.shape)
            action = action + noise
        action = action.numpy()
        return action.clip(self.env.action_space.low, self.env.action_space.high).squeeze()

    @torch.no_grad()
    def play_step(self, policy_network: torch.nn.Module, action_noise: float = 0.0) -> float:
        # Get action with exploration noise
        action = self.get_action(policy_network, action_noise)
        # take action step in the environment
        new_state, reward, done, _ = self.env.step(action)
        # Create experience object
        exp = Experience(self.state, action, reward, done, new_state)
        # add experience to replay buffer
        self.replay_buffer.append(exp)
        # Update state of agent with the newly experienced state
        self.state = new_state
        # reset environment if done
        if done:
            self.reset()
        return reward
