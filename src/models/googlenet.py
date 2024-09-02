import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models.googlenet import GoogLeNet


class GoogLeNet(GoogLeNet):
    def __init__(self):
        # Initialize the inherited GoogLeNet class
        super(GoogLeNet, self).__init__(aux_logits=False)
        
        # Modify the first convolutional layer to accept a single channel input (grayscale)
        self.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        
        # Replace the final fully connected layer with one that has num_classes=5 outputs
        self.fc = nn.Linear(self.fc.in_features, 6)
