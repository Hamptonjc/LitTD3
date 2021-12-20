# Imports
import torch
import torchvision

class PolicyNetwork(torch.nn.Module):
    
    def __init__(self, action_dim, max_action):
        super().__init__()
        #Conv
        self.resnet = torchvision.models.quantization.resnet18(pretrained=True)
        self.resnet.fc = torch.nn.Sequential(torch.nn.Linear(512,128), torch.nn.ReLU(),
                                          torch.nn.Linear(128,64), torch.nn.ReLU(),
                                          torch.nn.Linear(64, action_dim))
        self.max_action = max_action
        
    def forward(self, x):
        x = self.resnet(x)
        return x
    
    
    
class QNetworks(torch.nn.Module):
    def __init__(self, action_dim):
        super().__init__()
        # Conv
        self.resnet = torchvision.models.quantization.resnet18(pretrained=True)
        self.resnet = torch.Sequential(*list(self.resnet.children())[:-3])
        # Q-Net 1
        self.qnet1 = torch.nn.Sequential(torch.nn.Linear(512 + action_dim, 256), torch.nn.ReLU(),
                                         torch.nn.Linear(256, 64), torch.nn.ReLU(),
                                         torch.nn.Linear(64, 1))
        # Q-Net 2
        self.qnet2 = torch.nn.Sequential(torch.nn.Linear(512 + action_dim, 256), torch.nn.ReLU(),
                                         torch.nn.Linear(256, 64), torch.nn.ReLU(),
                                         torch.nn.Linear(64, 1))   
            
    def forward(self, state, action):
        # reduce state to 1-dimensional tensor
        state = self.resnet(state)
        state = state.view(-1, 512)
        # Concatenate state vector and action vector
        state_action = torch.cat([state, action], 1)
        # give state-action vector to each Q-network
        q1 = self.qnet1(state_action)
        q2 = self.qnet2(state_action)
        return q1, q2
    
    def get_Q1(self, state, action):
        # reduce state to 1-dimensional tensor
        state = self.resnet(state)
        state = state.view(-1, 512)
        # Concatenate state feature vector and action vector
        state_action = torch.cat([state, action], 1)
        # give state-action vector to Q-network
        q1 = self.qnet1(state_action)
        return q1
 
    def get_Q2(self, state, action):
        # reduce state to 1-dimensional tensor
        state = self.resnet(state)
        state = state.view(-1, 512)
        # Concatenate state vector and action vector
        state_action = torch.cat([state, action], 1)
        # give state-action vector to Q-network
        q2 = self.qnet2(state_action)
        return q2
 