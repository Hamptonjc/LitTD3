from typing import Dict
import copy
import torch
import pytorch_lightning as pl
from .networks import PolicyNetwork, QNetworks

#TODO Debug

class LitTD3(pl.LightningModule):


    def __init__(self, config: object):
        super().__init__()
        self.config = config
        # Networks
        self.policy = PolicyNetwork()
        self.qnets = QNetworks()
        self.target_policy = copy.deepcopy(self.policy)
        self.target_qnets = copy.deepcopy(self.qnets)
        # Populate replay buffer
        self.populate_buffer(config.WARM_START_STEPS)
        # Settings
        self.automatic_optimization = False

    def populate_buffer(self, steps: int) -> None:
        for _ in range(steps):
            self.datamodule.agent.play_step(self.policy, self.config.EXPLORATION_NOISE)


    def q_loss(self, current_Q1, current_Q2, target_Q):
        return torch.nn.functional.mse_loss(current_Q1, target_Q) + torch.nn.functional.mse_loss(current_Q2, target_Q)

    def policy_loss(self, states: torch.Tensor) -> torch.Tensor:
        return -self.qnets.get_Q1(states, self.policy(states)).mean()


    def configure_optimizers(self):
        policy_opt = torch.optim.Adam(self.policy.parameters(), lr=self.config.POLICY_LR)
        qnets_opt = torch.optim.Adam(self.qnets.parameters(), lr=self.config.QNETS_LR)
        return policy_opt, qnets_opt


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.target_policy(x)


    def on_train_batch_start(self, batch: torch.Tensor, batch_idx: int) -> None:
        # Create new experience
        reward = self.datamodule.agent.play_step(self.policy, self.config.EXPLORATION_NOISE)
        self.log('reward', reward, on_step=True, prog_bar=True, logger=True) 


    def training_step(self, batch: torch.Tensor, batch_idx: int) -> Dict[str, torch.Tensor] :
        # Get optimizers
        policy_opt, qnets_opt = self.optimizers()
        # Unpack batch
        states, actions, rewards, dones, next_states = batch

        with torch.no_grad():
            # select action with target policy and apply clipped noise
            noise = torch.normal(0, self.config.POLICY_NOISE, size=actions.shape).clamp(-self.config.POLICY_NOISE_CLIP,
                                                                                            self.config.POLICY_NOISE_CLIP)
            next_actions = (self.target_policy(next_states) + noise).clamp(self.env.action_space.low,
                                                                        self.env.action_space.high)
            # compute target Q values
            target_Q1, target_Q2 = self.target_qnets(next_states, next_actions)
            # take minimum of Q1 & Q2
            target_Q = torch.min(target_Q1, target_Q2)
            target_Q = rewards + self.conifg.DISCOUNT * (1 - dones) * target_Q

        # compute current Q values
        current_Q1, current_Q2 = self.qnets(states, actions)
        # calculate loss for current qnets
        qnet_loss = self.q_loss(current_Q1, target_Q, current_Q2, target_Q)
        # Optimize qnets
        qnets_opt.zero_grad()
        self.manual_backward(qnet_loss)
        qnets_opt.step()

        # Update policy every n steps
        if (batch_idx + 1) % self.policy_update_steps == 0:
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

        # log
        self.log('qnet_loss', qnet_loss, on_step=True, prog_bar=False, logger=True) 
        self.log('policy_loss', policy_loss, on_step=True, prog_bar=False, logger=True) 
        return {'qnet_loss': qnet_loss, 'policy_loss': policy_loss}