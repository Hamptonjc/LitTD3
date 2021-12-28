# inspired from -> https://www.pytorchlightning.ai/blog/en-lightning-reinforcement-learning-building-a-dqn-with-pytorch-lightning

# Imports
import torch
import gym
import numpy as np
from gymDataset import Experience, ReplayBuffer

class Agent:

    def __init__(self, env: gym.Env, replay_buffer: ReplayBuffer) -> None:
        self.env = env
        self.replay_buffer = replay_buffer
        self.state = self.env.reset()

    def reset(self) -> None:
        self.state = self.env.reset()

    def get_action(self, policy_network: torch.nn.Module, noise: float) -> np.array:
        # Convert state from numpy array to torch tensor
        state = torch.tensor(self.state)
        # get action from policy network
        action = policy_network(state)
        if noise != 0.0:
            noise = torch.normal(0, noise, size=action.shape)
            action = action + noise
        return action.clamp(self.env.action_space.low, self.env.action_space.high).numpy()

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
