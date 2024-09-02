import torch
import torch.nn as nn
import torchvision.models as models

class AlexNet(nn.Module):
    def __init__(self):
        super(AlexNet, self).__init__()
        # Load the AlexNet architecture
        self.alexnet = models.alexnet(pretrained=False)
        
        # Modify the first convolutional layer to accept 1 input channel instead of 3
        self.alexnet.features[0] = nn.Conv2d(1, 64, kernel_size=11, stride=4, padding=2)
        
        # Modify the final fully connected layer to output `num_classes=5` classes
        self.alexnet.classifier[6] = nn.Linear(self.alexnet.classifier[6].in_features, 6)

    def forward(self, x):
        return self.alexnet(x)
