import os
import numpy as np
import torch as T
import torch.nn as nn
from torch.nn.modules import linear
import torch.optim as optim
from torch.distributions.categorical import Categorical

class CriticNetwork(nn.Module):
    def __init__(self, input_dims, alpha, fc1_dims=256, fc2_dims=256, chkpt_dir='tmp/ppo'):
        super(CriticNetwork, self).__init__()

        self.checkpoint_file = os.path.join(chkpt_dir, 'critic_torch_ppo')
        self.critic = nn.Sequential(
            nn.Linear(128,256),
            nn.ReLU(),
            nn.Linear(256,128),
            nn.ReLU(),
            nn.Linear(128,64),
            nn.ReLU(),
            nn.Linear(64,1)
        )

        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)
    
    def forward(self, state):
        value = self.critic(state)
        return value

    def save_checkpoint(self, episode):
        T.save(self.state_dict(), self.checkpoint_file + episode)

    def load_checkpoint(self):
        self.load_state_dict(T.load(self.checkpoint_file))
