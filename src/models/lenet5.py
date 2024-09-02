import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch.nn.utils as torch_utils

class LeNet5(nn.Module):

    # network structure
    def __init__(self, in_channels):
        super(LeNet5, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=6, kernel_size=3, padding=2)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=in_channels, kernel_size=3)
        self.fc1 = nn.Linear(63*63*in_channels)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 6)

    def forward(self, x):
        x = x.squeeze(2)
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv2(x)), (2, 2))
        x = x.view(x.size(0), -1)
        
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        
        return x
        