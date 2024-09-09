import torch
import torch.nn as nn
from torchvision.models.resnet import ResNet, BasicBlock, Bottleneck


class ResNet18(ResNet):
    def __init__(self):
        # Initialize the inherited ResNet class with the BasicBlock and layers configuration for ResNet-18
        super(ResNet18, self).__init__(BasicBlock, [2, 2, 2, 2])
        
        self.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)

        # Replace the final fully connected layer with one that has `num_classes=6` outputs
        self.fc = nn.Linear(self.fc.in_features, 6)

class ResNet34(ResNet):
    def __init__(self):
        # Initialize the inherited ResNet class with the BasicBlock and layers configuration for ResNet-34
        super(ResNet34, self).__init__(BasicBlock, [3, 4, 6, 3])
        
        self.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)

        # Replace the final fully connected layer with one that has `num_classes=6` outputs
        self.fc = nn.Linear(self.fc.in_features, 6)

class ResNet50(ResNet):
    def __init__(self,):
        # Initialize the inherited ResNet class with the Bottleneck block and layers configuration for ResNet-50
        super(ResNet50, self).__init__(Bottleneck, [3, 4, 6, 3])

        self.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        
        # Replace the final fully connected layer with one that has `num_classes=6` outputs
        self.fc = nn.Linear(self.fc.in_features, 6)

class ResNet101(ResNet):
    def __init__(self):
        # Initialize the inherited ResNet class with the Bottleneck block and layers configuration for ResNet-101
        super(ResNet101, self).__init__(Bottleneck, [3, 4, 23, 3])

        self.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        
        # Replace the final fully connected layer with one that has `num_classes=6` outputs
        self.fc = nn.Linear(self.fc.in_features, 6)

class ResNet152(ResNet):
    def __init__(self):
        # Initialize the inherited ResNet class with the Bottleneck block and layers configuration for ResNet-152
        super(ResNet152, self).__init__(Bottleneck, [3, 8, 36, 3])

        self.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)

        # Replace the final fully connected layer with one that has `num_classes=6` outputs
        self.fc = nn.Linear(self.fc.in_features, 6)
