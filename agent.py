# Imports
import collections
from typing import Tuple, NamedTuple
import torch
import numpy as np
from master_config import MasterConfig


# Named tuple for storing experience steps gathered in training
class Experience(NamedTuple):
    state: np.array
    action: float
    reward: float
    done: bool
    next_state: np.array
    

class ReplayBuffer:
    """
    ReplayBuffer: Data structure for storing
        past experiences allowing the agent to learn from them
    Args:
        capacity: size of the buffer
    """

    def __init__(self, capacity: int) -> None:
        self.buffer = collections.deque(maxlen=capacity)

    def __len__(self) -> None:
        return len(self.buffer)

    def append(self, experience: Experience) -> None:
        self.buffer.append(experience)

    def sample(self, batch_size: int) -> Tuple[np.array]:
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        (states, actions, rewards,
            dones, next_states) = zip(*[self.buffer[idx] for idx in indices])
        return (np.array(states, dtype=np.float32),
                np.array(actions, dtype=np.float32),
                np.array(rewards, dtype=np.float32),
                np.array(dones, dtype=int),
                np.array(next_states, dtype=np.float32))


class TD3Agent:
    """
    TD3Agent: Handles interfacing with the environment following TD3.
    
    env (any): An OpenAI-Gym-like environment.
    replay_buffer (ReplayBuffer): A buffer to store experiences.
    """

    def __init__(self, config: MasterConfig,
                env: any, replay_buffer: ReplayBuffer) -> None:
        self.config = config
        self.env = env
        self.replay_buffer = replay_buffer
        self.state = self.env.reset(seed=config.RANDOM_SEED)

    def reset(self) -> None:
        self.state = self.env.reset(seed=self.config.RANDOM_SEED)

    def get_action(self, actor_network: torch.nn.Module, 
                                noise: float) -> np.array:
        # Convert state from numpy array to torch tensor
        device = next(actor_network.parameters()).device 
        if len(self.env.observation_space.shape) > 1:
            state = torch.tensor(
                (self.state.copy()).astype(np.float32))\
                    .permute(2,1,0).unsqueeze(0).to(device)
        else:
            state = torch.tensor((self.state.copy()).astype(np.float32))\
                                    .unsqueeze(0).to(device)
        # get action from actor network
        action = actor_network(state)
        noise = torch.normal(0, noise, 
                    size=action.shape).to(device)
        action = action + noise
        action = action.cpu().numpy()
        return action.clip(self.env.action_space.low,
                            self.env.action_space.high).squeeze()

    @torch.no_grad()
    def play_step(self, actor_network: torch.nn.Module,
                    action_noise: float = 0.0) -> Tuple[float, int]:
        # Get action with exploration noise
        action = self.get_action(actor_network, action_noise)
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
        return reward, done

    def play_random_step(self) -> None:
        # Get action with exploration noise
        #actions = []
        #for l, h in zip(self.env.action_space.low,
        #                self.env.action_space.high):
        #    actions.append(np.random.uniform(l,h,1))
        action = self.env.action_space.sample()
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
