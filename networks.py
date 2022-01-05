# Imports
from typing import Union
import copy
import torch
import torchvision

class PolicyNetwork(torch.nn.Module):
    
    def __init__(self, action_space_len: int, state_dims: int,
                 state_space_len: Union[int, None]) -> None:
        super().__init__()
        # Use conv if state is multi-dimensional
        if state_dims > 1:
            self.net = torchvision.models.quantization.mobilenet_v2(pretrained=True)
            self.net.classifier = torch.nn.Sequential(torch.nn.Dropout(p=0.2),
                                                      torch.nn.Linear(1280,256),
                                                      torch.nn.ReLU(),
                                                      torch.nn.Linear(256, action_space_len),
                                                      torch.nn.Tanh())
        # else use a fully-connected network
        else:
            self.net = torch.nn.Sequential(torch.nn.Linear(state_space_len,256),
                                           torch.nn.ReLU(),
                                           torch.nn.Linear(256,256),
                                           torch.nn.ReLU(),
                                           torch.nn.Linear(256, action_space_len),
                                           torch.nn.Tanh())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.net(x)
        return x


class QNetworks(torch.nn.Module):

    def __init__(self, action_space_len: int, state_dims: int,
                 state_space_len: Union[int, None]) -> None:
        super().__init__()
        # use conv if state is multi-dimensional
        if state_dims > 1:
            self.qnet1_conv = torchvision.models.quantization.mobilenet_v2(pretrained=True)
            self.qnet1_conv = torch.nn.Sequential(*list(self.qnet1_conv.children())[:-3],
                                                  torch.nn.AdaptiveAvgPool2d((1,1)))
            self.qnet2_conv = copy.deepcopy(self.qnet1_conv)
            # Q-Net 1
            self.qnet1 = torch.nn.Sequential(torch.nn.Dropout(p=0.2), 
                                             torch.nn.Linear(1280 + action_space_len, 256),
                                             torch.nn.ReLU(),
                                             torch.nn.Linear(256, 1))
            # Q-Net 2
            self.qnet2 = copy.deepcopy(self.qnet1)
        # else use FCN if state dims == 1
        else:
            self.qnet1 = torch.nn.Sequential(torch.nn.Linear(state_space_len + \
                                                             action_space_len,256),
                                             torch.nn.ReLU(),
                                             torch.nn.Linear(256,256),
                                             torch.nn.ReLU(),
                                             torch.nn.Linear(256, 1),
                                             torch.nn.Tanh())
            self.qnet2 = copy.deepcopy(self.qnet1)


    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        q1 = self.get_Q1(state, action)
        q2 = self.get_Q2(state, action)
        return q1, q2
    
    def get_Q1(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        # reduce state to 1-dimensional tensor
        if hasattr(self, 'qnet1_conv'):
            state = self.qnet1_conv(state)
            state = state.view(-1, 1280)
        # Concatenate state feature vector and action vector
        state_action = torch.cat([state, action], 1)
        # give state-action vector to Q-network
        q1 = self.qnet1(state_action)
        return q1
 
    def get_Q2(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        # reduce state to 1-dimensional tensor
        if hasattr(self, 'qnet2_conv'):
            state = self.qnet2_conv(state)
            state = state.view(-1, 1280)
        # Concatenate state vector and action vector
        state_action = torch.cat([state, action], 1)
        # give state-action vector to Q-network
        q2 = self.qnet2(state_action)
        return q2
 