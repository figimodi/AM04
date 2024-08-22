import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models.resnet import ResNet, BasicBlock, Bottleneck

class ResNet18(ResNet):
    def __init__(self):
        # Initialize the inherited ResNet class with the BasicBlock and layers configuration for ResNet-18
        super(ResNet18, self).__init__(BasicBlock, [2, 2, 2, 2])
        
        # Optionally load the pretrained weights for resnet18
        # pretrained_model = models.resnet18(pretrained=True)
        # self.load_state_dict(pretrained_model.state_dict())

        # Replace the final fully connected layer with one that has `num_classes=5` outputs
        self.fc = nn.Linear(self.fc.in_features, 5)

class ResNet34(ResNet):
    def __init__(self):
        # Initialize the inherited ResNet class with the BasicBlock and layers configuration for ResNet-34
        super(ResNet34, self).__init__(BasicBlock, [3, 4, 6, 3])
        
        # Optionally load the pretrained weights for resnet34
        pretrained_model = models.resnet34(pretrained=True)
        self.load_state_dict(pretrained_model.state_dict())

        # Replace the final fully connected layer with one that has `num_classes=5` outputs
        self.fc = nn.Linear(self.fc.in_features, 5)

class ResNet50(ResNet):
    def __init__(self,):
        # Initialize the inherited ResNet class with the Bottleneck block and layers configuration for ResNet-50
        super(ResNet50, self).__init__(Bottleneck, [3, 4, 6, 3])
        
        # Optionally load the pretrained weights for resnet50
        pretrained_model = models.resnet50(pretrained=True)
        self.load_state_dict(pretrained_model.state_dict())

        # Replace the final fully connected layer with one that has `num_classes=5` outputs
        self.fc = nn.Linear(self.fc.in_features, 5)

class ResNet101(ResNet):
    def __init__(self):
        # Initialize the inherited ResNet class with the Bottleneck block and layers configuration for ResNet-101
        super(ResNet101, self).__init__(Bottleneck, [3, 4, 23, 3])
        
        # Optionally load the pretrained weights for resnet101
        pretrained_model = models.resnet101(pretrained=True)
        self.load_state_dict(pretrained_model.state_dict())

        # Replace the final fully connected layer with one that has `num_classes=5` outputs
        self.fc = nn.Linear(self.fc.in_features, 5)

class ResNet152(ResNet):
    def __init__(self):
        # Initialize the inherited ResNet class with the Bottleneck block and layers configuration for ResNet-152
        super(ResNet152, self).__init__(Bottleneck, [3, 8, 36, 3])
        
        # Optionally load the pretrained weights for resnet152
        # pretrained_model = models.resnet152(pretrained=True)
        # self.load_state_dict(pretrained_model.state_dict())

        # Replace the final fully connected layer with one that has `num_classes=5` outputs
        self.fc = nn.Linear(self.fc.in_features, 5)

