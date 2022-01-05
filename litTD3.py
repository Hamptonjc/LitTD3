# Imports
from typing import Union, Tuple
import copy, os
import torch
import pytorch_lightning as pl
from networks import PolicyNetwork, QNetworks

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

class LitTD3(pl.LightningModule):


    def __init__(self, config: object, action_space_len: int, 
                 state_dims: int, state_space_len: Union[int, None]) -> None:
        super().__init__()
        self.config = config
        self.n_iterations = 0
        # Networks
        self.policy = PolicyNetwork(action_space_len, state_dims, state_space_len)
        self.qnets = QNetworks(action_space_len, state_dims, state_space_len)
        self.target_policy = copy.deepcopy(self.policy)
        self.target_qnets = copy.deepcopy(self.qnets)
        # Settings
        self.automatic_optimization = False


    def on_fit_start(self) -> None:
        self.populate_buffer(self.config.WARM_START_STEPS)


    def populate_buffer(self, steps: int) -> None:
        for _ in range(steps):
            self.trainer.datamodule.agent.play_random_step()


    def q_loss(self, current_Q1: torch.Tensor,
               current_Q2: torch.Tensor, target_Q: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.mse_loss(current_Q1, target_Q) + \
                torch.nn.functional.mse_loss(current_Q2, target_Q)


    def policy_loss(self, states: torch.Tensor) -> torch.Tensor:
        return -self.qnets.get_Q1(states, self.policy(states)).mean()


    def configure_optimizers(self) -> Tuple[torch.optim.Optimizer]:
        policy_opt = torch.optim.Adam(self.policy.parameters(), lr=self.config.POLICY_LR)
        qnets_opt = torch.optim.Adam(self.qnets.parameters(), lr=self.config.QNETS_LR)
        return policy_opt, qnets_opt


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.target_policy(x)


    def on_train_batch_start(self, *args, **kwargs) -> None:
        # Do live validation if time to
        if (self.n_iterations + 1) % self.config.VAL_CHECK_INTERVAL == 0:
            self.live_validation()
        # Create new experience
        self.trainer.datamodule.agent.play_step(self.policy, self.config.EXPLORATION_NOISE)


    def training_step(self, batch: torch.Tensor, batch_idx: int) -> None:
        # Get optimizers
        policy_opt, qnets_opt = self.optimizers()
        # Unpack batch
        states, actions, rewards, dones, next_states = batch
        # Get the target Q values from target Q-Networks
        with torch.no_grad():
            # select action with target policy and apply clipped noise
            noise = torch.normal(0, self.config.POLICY_NOISE,
                                 size=actions.shape, device=device).clamp(-self.config.POLICY_NOISE_CLIP,
                                                                           self.config.POLICY_NOISE_CLIP)
            next_actions = (self.target_policy(next_states) + noise).clamp(
                torch.tensor(self.trainer.datamodule.env.action_space.low).to(device),
                torch.tensor(self.trainer.datamodule.env.action_space.high).to(device))
            # compute target Q values
            target_Q1, target_Q2 = self.target_qnets(next_states, next_actions)
            # take minimum of Q1 & Q2
            target_Q = torch.min(target_Q1, target_Q2)
            target_Q = rewards + self.config.DISCOUNT * (1 - dones) * target_Q
        # compute current Q values
        current_Q1, current_Q2 = self.qnets(states, actions)
        # calculate loss for current qnets
        qnet_loss = self.q_loss(current_Q1, current_Q2, target_Q)
        # Optimize qnets
        qnets_opt.zero_grad()
        self.manual_backward(qnet_loss)
        qnets_opt.step()
        # Update policy every n steps
        if (self.n_iterations + 1) % self.config.POLICY_DELAY == 0:
            # Calculate policy loss
            policy_loss = self.policy_loss(states)
            # Optimize policy
            policy_opt.zero_grad()
            self.manual_backward(policy_loss)
            policy_opt.step()
            # Update target qnets network
            for param, target_param in zip(self.qnets.parameters(), self.target_qnets.parameters()):
                target_param.data.copy_(self.config.TAU * param.data + \
                                        (1 - self.config.TAU) * target_param.data)
            # Update target policy network
            for param, target_param in zip(self.policy.parameters(), self.target_policy.parameters()):
                target_param.data.copy_(self.config.TAU * param.data + \
                                        (1 - self.config.TAU) * target_param.data)
            # Log policy network loss
            self.log('policy_loss', policy_loss, on_step=True, prog_bar=False, logger=True) 
        # log
        self.log('qnet_loss', qnet_loss, on_step=True, prog_bar=False, logger=True) 
        self.n_iterations += 1


    def live_validation(self) -> None:
        # Initalize episode counter & rewards list
        episode_count = 0
        rewards = []
        # Reset environment
        self.trainer.datamodule.agent.reset()
        # While number of validation episodes hasn't been reached
        while episode_count < self.config.VAL_EPISODES:
            # play step
            reward, done = self.trainer.datamodule.agent.play_step(self.policy, 0.0)
            # add reward to rewards list
            rewards.append(reward)
            # Increase episode count if done
            episode_count += done
        # log the average reward over n episodes
        self.log('reward', sum(rewards)/len(rewards), on_step=True, prog_bar=True, logger=True)
