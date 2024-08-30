import torch
import torch.nn as nn
import torchvision
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.models import resnet50
from torchvision.models.detection.backbone_utils import BackboneWithFPN

class MyFasterRCNN(nn.Module):
    def __init__(self, num_classes=5):
        super(MyFasterRCNN, self).__init__()

        # Load a pre-trained ResNet50 backbone
        backbone = resnet50(pretrained=True)

        # Modify the first convolutional layer to accept 1 input channel instead of 3
        # Original layer: Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        backbone.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        
        # Extract the layers except for the final fully connected layer
        # The backbone must return feature maps for each layer in the feature pyramid
        backbone_with_fpn = BackboneWithFPN(
            backbone, 
            return_layers={'layer1': '0', 'layer2': '1', 'layer3': '2', 'layer4': '3'},
            in_channels_list=[256, 512, 1024, 2048],
            out_channels=256
        )

        # Define the Region Proposal Network (RPN) anchor generator
        rpn_anchor_generator = AnchorGenerator(
            sizes=((16, 64, 128, 256, 512),),  # Added a larger size
            aspect_ratios=((0.1, 0.5, 1.0, 2.0, 4.0),) * 5  # Added a larger aspect ratio for extremely tall objects
        )

        # Define the ROI Pooling feature extractor
        roi_pooler = torchvision.ops.MultiScaleRoIAlign(
            featmap_names=['0', '1', '2', '3'],
            output_size=7,
            sampling_ratio=2
        )

        # Create the Faster R-CNN model
        self.model = FasterRCNN(
            backbone_with_fpn,
            num_classes=num_classes,
            rpn_anchor_generator=rpn_anchor_generator,
            box_roi_pool=roi_pooler
        )

    def forward(self, x):
        if len(x) == 2:   
            return self.model(x[0], x[1])
        else:
            return self.model(x)
        