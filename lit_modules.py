# Imports
import copy
from typing import Tuple, Optional, Any
import gym
import torch
import pytorch_lightning as pl
from agent import *
from master_config import MasterConfig
from datasets import OneDimStateGymDataset, TwoDimStateGymDataset
from networks import OneDimStateActorNetwork, OneDimStateCriticNetworks, TwoDimStateActorNetwork, TwoDimStateCriticNetworks


#####################################################################
#   TD3 Lightning Module
#####################################################################

class TD3LitModule(pl.LightningModule):
    
    """
    LitTD3: A PyTorch Lightning Module implementing the main training logic behind TD3
    
    config (MasterConfig): Master configuration instance.
    """

    def __init__(self, config: MasterConfig) -> None:
        super().__init__()
        # add MasterConfig as a instance attribute
        self.config = config
        # Create counter to count the number of iterations
        self.n_iterations = 0
        # Create networks
        self.__create_networks()
        # Turn off automatic .backwards() calls
        self.automatic_optimization = False


    def on_fit_start(self) -> None:
        """ Before training starts, play n steps to fill buffer """
        self.__populate_buffer(int(self.config.WARM_START_STEPS))


    def critic_loss(self, current_Q1: torch.Tensor,
                    current_Q2: torch.Tensor,
                    target_Q: torch.Tensor) -> torch.Tensor:
        """ Loss function for critic networks. """
        return torch.nn.functional.mse_loss(
                current_Q1, target_Q) + \
                torch.nn.functional.mse_loss(
                    current_Q2, target_Q)


    def actor_loss(self, state: torch.Tensor) -> torch.Tensor:
        """ Loss function for actor network. """
        return -self.critics.get_critic1(
                    state, self.actor(state)).mean()


    def configure_optimizers(self) -> Tuple[torch.optim.Optimizer]:
        """ Configure gradient optimization algorithms. """
        actor_opt = torch.optim.Adam(self.actor.parameters(),
                                    lr=self.config.ACTOR_LR)
        critics_opt = torch.optim.Adam(self.critics.parameters(),
                                        lr=self.config.CRITICS_LR)
        return actor_opt, critics_opt


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.target_actor(x)


    def on_train_batch_start(self, batch: Any,
        batch_idx: int, unused: int = 0) -> Optional[int]:
        """ Function called before each training data batch. """
        # Do validation if time to
        if self.n_iterations % self.config.VAL_CHECK_INTERVAL == 0:
            self.validation()
        # Create new experience
        self.trainer.datamodule.agent.play_step(
            self.actor, self.config.EXPLORATION_NOISE)


    def training_step(self, batch: torch.Tensor, batch_idx: int) -> None:
        """ A single training step of TD3 """
        # Get optimizers
        actor_opt, critics_opt = self.optimizers()
        # Unpack batch
        state, action, reward, done, next_state = batch
        # Get the target Q-values from target critic networks
        with torch.no_grad():
            # select action with target actor and apply clipped noise
            noise = torch.normal(0, self.config.ACTOR_NOISE,
                size=action.shape, device=self.device).clamp(
                                        -self.config.ACTOR_NOISE_CLIP,
                                        self.config.ACTOR_NOISE_CLIP)
            next_action = (self.target_actor(next_state) + noise)\
                .clamp(torch.tensor(
                            self.config.MIN_ACTION).to(self.device),
                        torch.tensor(
                            self.config.MAX_ACTION).to(self.device))
            # compute target Q-values
            target_Q1, target_Q2 = self.target_critics(
                                        next_state, next_action)
            # take minimum of Q1 & Q2
            target_Q = torch.min(target_Q1, target_Q2)
            target_Q = reward + self.config.DISCOUNT\
                                * (1 - done) * target_Q
        # compute current Q-values
        current_Q1, current_Q2 = self.critics(state, action)
        # calculate loss for current critics
        critic_loss = self.critic_loss(current_Q1,
                        current_Q2, target_Q)
        # Optimize critics
        critics_opt.zero_grad()
        self.manual_backward(critic_loss)
        critics_opt.step()
        # Update actor every n steps
        if self.n_iterations % self.config.ACTOR_DELAY == 0:
            # Calculate actor loss
            actor_loss = self.actor_loss(state)
            # Optimize actor
            actor_opt.zero_grad()
            self.manual_backward(actor_loss)
            actor_opt.step()
            # Update target critics network
            for param, target_param in zip(self.critics.parameters(),
                                    self.target_critics.parameters()):
                target_param.data.copy_(self.config.TAU * param.data +\
                            (1 - self.config.TAU) * target_param.data)
            # Update target actor network
            for param, target_param in zip(self.actor.parameters(),
                                    self.target_actor.parameters()):
                target_param.data.copy_(self.config.TAU * param.data +\
                            (1 - self.config.TAU) * target_param.data)
            # Log actor network loss
            self.log('Actor Loss', actor_loss,
                on_step=True, prog_bar=False, logger=True) 
        # log
        self.log('Critic Loss', critic_loss,
            on_step=True, prog_bar=False, logger=True) 
        self.n_iterations += 1


    def validation(self) -> None:
        """ Perform full episodes without training for evaluation of the policy """
        # Initalize episode counter, rewards list, & return list
        episode_count = 0
        rewards = []
        returns = []
        _return = 0.
        # Reset environment
        self.trainer.datamodule.agent.reset()
        # While number of validation episodes hasn't been reached
        while episode_count < self.config.VAL_EPISODES:
            # play step
            reward, done = self.trainer.datamodule.agent.play_step(
                                                    self.actor, 0.0)
            # add reward to rewards list
            rewards.append(reward)
            # add reward to return
            _return += reward
            if done:
                returns.append(_return)
                _return = 0.
            # Increase episode count if done
            episode_count += done
        # log the average reward and average return over n episodes
        self.log('Average Reward', sum(rewards)/len(rewards), 
                    on_step=True, prog_bar=True, logger=True)
        self.log('Average Return', sum(returns)/len(returns),
                    on_step=True, prog_bar=True, logger=True)


    def __populate_buffer(self, steps: int) -> None:
        """ Play random steps to fill buffer """
        for _ in range(steps):
            self.trainer.datamodule.agent.play_random_step()


    def __create_networks(self) -> None:
        """ Create networks based on the state dimension. """
        if self.config.STATE_DIMS == 1:
            self.actor = OneDimStateActorNetwork(self.config)
            self.critics = OneDimStateCriticNetworks(self.config)
        elif self.config.STATE_DIMS == 2:
            self.actor = TwoDimStateActorNetwork(self.config)
            self.critics = TwoDimStateCriticNetworks(self.config)
        else:
            raise NotImplementedError()
        self.target_actor = copy.deepcopy(self.actor)
        self.target_critics = copy.deepcopy(self.critics)


#####################################################################
#   TD3 Lightning DataModule
#####################################################################

class TD3LitDataModule(pl.LightningDataModule):

    def __init__(self, config: MasterConfig) -> None:
        self.config = config
        self.replay_buffer = ReplayBuffer(config.BUFFER_SIZE)
        self.prepare_data_per_node = False
        self._log_hyperparams = False


    def prepare_data(self) -> None:
        pass


    def setup(self, stage: str=None) -> None:
        # Environment
        if self.config.ENVIRONMENT_API == 'gym':
            self.__setup_gym_env()
        
        # **********************************
        # * Add new environment setup here *
        # **********************************

        else:
            raise NotImplementedError(("Environment API"
                                    f"{self.config.ENVIRONMENT_API}"
                                    "currently not supported."))
        # Agent
        self.agent = TD3Agent(self.config, self.env, self.replay_buffer)


    def train_dataloader(self) -> torch.utils.data.DataLoader:
        dataloader = torch.utils.data.DataLoader(self.dataset,
                        batch_size=self.config.BATCH_SIZE,
                        num_workers=0)
        return dataloader


    def __setup_gym_env(self) -> None:
        # Create Environment
        self.env = gym.make(**self.config.ENVIRONMENT_SETTINGS)
        # Add/adjust configs to environment specifics
        self.config.ACTION_SPACE_LEN = self.env.action_space.shape[0]
        self.config.STATE_DIMS = len(self.env.observation_space.shape)
        self.config.STATE_SPACE_LEN = self.env.observation_space.shape[0]
        self.config.MAX_ACTION = self.env.action_space.high
        self.config.MIN_ACTION = self.env.action_space.low
        self.config.ACTOR_NOISE *= self.env.action_space.high[0].item()
        self.config.ACTOR_NOISE_CLIP *= self.env.action_space.high[0].item()
        # Instatiate Torch dataset based on environment
        if self.config.STATE_DIMS == 1:
            self.dataset = OneDimStateGymDataset(
                self.replay_buffer, self.config.BATCH_SIZE)
        elif self.config.STATE_DIMS == 2:
            self.dataset = TwoDimStateGymDataset(
                self.replay_buffer, self.config.BATCH_SIZE)

    # ****************************************
    # * Add new environment setup func. here *
    # ****************************************
 