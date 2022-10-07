# Imports
import copy
import torch, torchvision
from master_config import MasterConfig


#####################################################################
#   Actor Networks
#####################################################################
class ActorNetworkBase(torch.nn.Module):

    """
    ActorNetworkBase: Base class which all other 
                        actor networks derive from.

    The actor network maps a state to
             an action (i.e. actor(state) = action)

    config (MasterConfig): Master configuration object
    """

    def __init__(self, config: MasterConfig) -> None:
        super().__init__()
        self.config = config


    def forward(self, state: torch.Tensor) -> torch.Tensor:
        action = self.network(state)
        return torch.tensor(self.config.MAX_ACTION).type_as(action) * action


class TwoDimStateActorNetwork(ActorNetworkBase):
    """
    TwoDimStateActorNetwork: An example actor network for a 
                    task with a 2-dimensional state space.

    config (MasterConfig): Master configuration object.
    """
    
    def __init__(self, config: MasterConfig) -> None:
        super().__init__(config)
        # Create network
        self.network = \
            torchvision.models.mobilenet_v2(
                            pretrained=config.PRETRAINED)
        self.network.classifier = torch.nn.Sequential(
                                    torch.nn.Dropout(p=0.2),
                                    torch.nn.Linear(1280,256),
                                    torch.nn.ReLU(),
                                    torch.nn.Linear(256, 
                                        config.ACTION_SPACE_LEN),
                                    torch.nn.Tanh())


class OneDimStateActorNetwork(ActorNetworkBase):
    """
    OneDimStateActorNetwork: An example actor network for a 
                    task with a 1-dimensional state space.

    config (MasterConfig): Master configuration object.
    """

    def __init__(self, config: MasterConfig) -> None:
        super().__init__(config)
 
        self.network = torch.nn.Sequential(
                            torch.nn.Linear(
                                config.STATE_SPACE_LEN, 256),
                            torch.nn.ReLU(),
                            torch.nn.Linear(256,256),
                            torch.nn.ReLU(),
                            torch.nn.Linear(
                                256, config.ACTION_SPACE_LEN),
                            torch.nn.Tanh())


#####################################################################
#   Critic Networks
#####################################################################
class CriticsNetworkBase(torch.nn.Module):

    """
    CriticsNetworkBase: Base class which all other 
                        critic networks derive from.

    The critic networks map a state and action to
             a Q-value (i.e. critic(state, action) = Q-value)

    config (MasterConfig): Master configuration object
    """

    def __init__(self, config: MasterConfig) -> None:
        super().__init__()
        self.config = config

    
    def forward(self, state: torch.Tensor,
                action: torch.Tensor) -> torch.Tensor:
        q1 = self.get_critic1(state, action)
        q2 = self.get_critic2(state, action)
        return q1, q2
 

    def get_critic1(self, state: torch.Tensor,
                    action: torch.Tensor) -> torch.Tensor:
        # Concatenate state feature vector and action vector
        state_action = torch.cat([state, action], 1)
        # give state-action vector to Q-network
        q = self.critic1(state_action)
        return q
 

    def get_critic2(self, state: torch.Tensor, 
                    action: torch.Tensor) -> torch.Tensor:
        # Concatenate state feature vector and action vector
        state_action = torch.cat([state, action], 1)
        # give state-action vector to Q-network
        q = self.critic2(state_action)
        return q
 

class TwoDimStateCriticNetworks(CriticsNetworkBase):

    """
    TwoDimStateCriticNetworks: An example actor network for a 
                    task with a 2-dimensional state space.

    config (MasterConfig): Master configuration object.
    """

    def __init__(self, config: MasterConfig) -> None:
        super().__init__(config)
        # CNN
        self.critic1_conv =\
            torchvision.models.mobilenet_v2(pretrained=True)
        self.critic1_conv = torch.nn.Sequential(
                *list(self.critic1_conv.children())[:-1],
                torch.nn.AdaptiveAvgPool2d((1,1)))
        self.critic2_conv = copy.deepcopy(self.critic1_conv)
        # Linear Network
        self.critic1 = torch.nn.Sequential(
                        torch.nn.Dropout(p=0.2), 
                        torch.nn.Linear(
                            1280 + config.ACTION_SPACE_LEN, 256),
                        torch.nn.ReLU(),
                        torch.nn.Linear(256, 1))
        self.critic2 = copy.deepcopy(self.critic1)

   
    def get_critic1(self, state: torch.Tensor,
                    action: torch.Tensor) -> torch.Tensor:
        # reduce state to 1-dimensional tensor
        state = self.critic1_conv(state)
        state = state.view(-1, 1280)
        q1 = super.get_critic1(state, action)
        return q1
 

    def get_critic2(self, state: torch.Tensor,
                    action: torch.Tensor) -> torch.Tensor:
        # reduce state to 1-dimensional tensor
        state = self.critic2_conv(state)
        state = state.view(-1, 1280)
        q2 = super.get_critic2(state, action)
        return q2
        

class OneDimStateCriticNetworks(CriticsNetworkBase):

    """
    OneDimStateCritcNetworks: An example of critic networks for a 
                    task with a 1-dimensional state space.

    config (MasterConfig): Master configuration object.
    """

    def __init__(self, config: MasterConfig) -> None:
        super().__init__(config)
        self.critic1 = torch.nn.Sequential(
            torch.nn.Linear(config.STATE_SPACE_LEN+ \
                            config.ACTION_SPACE_LEN,256),
            torch.nn.ReLU(),
            torch.nn.Linear(256,256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 1),
            torch.nn.Tanh())
        self.critic2 = copy.deepcopy(self.critic1)
