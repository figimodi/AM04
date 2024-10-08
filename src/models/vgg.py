import torch
import torch.nn as nn


class GrayVGG16(nn.Module):
    def __init__(self, in_dim = 1, num_classes = 6):
        super(GrayVGG16, self).__init__()
        # feature extraction part
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels = in_dim, out_channels = 64, kernel_size = 3, stride = 1, padding = 1),
            nn.ReLU(inplace = True),
            nn.Conv2d(in_channels = 64, out_channels = 64, kernel_size = 3, stride = 1, padding = 1)
        )
        self.pool1 = nn.Sequential(
            nn.ReLU(inplace = False),
            nn.MaxPool2d(kernel_size = 2, stride = 2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels = 64, out_channels = 128, kernel_size = 3, stride = 1, padding = 1),
            nn.ReLU(inplace = True),
            nn.Conv2d(in_channels = 128, out_channels = 128, kernel_size = 3, stride = 1, padding = 1)
        )
        self.pool2 = nn.Sequential(
            nn.ReLU(inplace = False),
            nn.MaxPool2d(kernel_size = 2, stride = 2)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels = 128, out_channels = 256, kernel_size = 3, stride = 1, padding = 1),
            nn.ReLU(inplace = True),
            nn.Conv2d(in_channels = 256, out_channels = 256, kernel_size = 3, stride = 1, padding = 1),
            nn.ReLU(inplace = True),
            nn.Conv2d(in_channels = 256, out_channels = 256, kernel_size = 3, stride = 1, padding = 1)
        )
        self.pool3 = nn.Sequential(
            nn.ReLU(inplace = False),
            nn.MaxPool2d(kernel_size = 2, stride = 2)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels = 256, out_channels = 512, kernel_size = 3, stride = 1, padding = 1),
            nn.ReLU(inplace = True),
            nn.Conv2d(in_channels = 512, out_channels = 512, kernel_size = 3, stride = 1, padding = 1),
            nn.ReLU(inplace = True),
            nn.Conv2d(in_channels = 512, out_channels = 512, kernel_size = 3, stride = 1, padding = 1)
        )
        self.pool4 = nn.Sequential(
            nn.ReLU(inplace = False),
            nn.MaxPool2d(kernel_size = 2, stride = 2)
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(in_channels = 512, out_channels = 512, kernel_size = 3, stride = 1, padding = 1),
            nn.ReLU(inplace = True),
            nn.Conv2d(in_channels = 512, out_channels = 512, kernel_size = 3, stride = 1, padding = 1),
            nn.ReLU(inplace = True),
            nn.Conv2d(in_channels = 512, out_channels = 512, kernel_size = 3, stride = 1, padding = 1)
        )
        self.pool5 = nn.Sequential(
            nn.ReLU(inplace = False),
            nn.AdaptiveAvgPool2d((7, 7))
        )
        # classification part
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(inplace = True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace = True),
            nn.Dropout(),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):                                   # shape: [B, 1, 224, 224]
        conv1 = self.conv1(x)                               # shape: [B, 64, 224, 224]
        pool1 = self.pool1(conv1)                           # shape: [B, 64, 112, 112]
        conv2 = self.conv2(pool1)                           # shape: [B, 128, 112, 112]
        pool2 = self.pool2(conv2)                           # shape: [B, 128, 56, 56]
        conv3 = self.conv3(pool2)                           # shape: [B, 256, 56, 56]
        pool3 = self.pool3(conv3)                           # shape: [B, 256, 28, 28]
        conv4 = self.conv4(pool3)                           # shape: [B, 512, 28, 28]
        pool4 = self.pool4(conv4)                           # shape: [B, 512, 14, 14]
        conv5 = self.conv5(pool4)                           # shape: [B, 512, 14, 14]
        pool5 = self.pool5(conv5)                           # shape: [B, 512, 7, 7]
        pool5 = pool5.view(x.size(0), -1)                   # shape: [B, 512 * 7 * 7]
        x = self.classifier(pool5)                          # shape: [B, 1000]
        return x

class GrayVGG16_BN(nn.Module):
    def __init__(self, in_dim = 1, num_classes = 5):
        super(GrayVGG16_BN, self).__init__()
        # feature extraction part
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels = in_dim, out_channels = 64, kernel_size = 3, stride = 1, padding = 1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace = True),
            nn.Conv2d(in_channels = 64, out_channels = 64, kernel_size = 3, stride = 1, padding = 1)
        )
        self.pool1 = nn.Sequential(
            nn.BatchNorm2d(64),
            nn.ReLU(inplace = False),
            nn.MaxPool2d(kernel_size = 2, stride = 2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels = 64, out_channels = 128, kernel_size = 3, stride = 1, padding = 1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace = True),
            nn.Conv2d(in_channels = 128, out_channels = 128, kernel_size = 3, stride = 1, padding = 1)
        )
        self.pool2 = nn.Sequential(
            nn.BatchNorm2d(128),
            nn.ReLU(inplace = False),
            nn.MaxPool2d(kernel_size = 2, stride = 2)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels = 128, out_channels = 256, kernel_size = 3, stride = 1, padding = 1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace = True),
            nn.Conv2d(in_channels = 256, out_channels = 256, kernel_size = 3, stride = 1, padding = 1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace = True),
            nn.Conv2d(in_channels = 256, out_channels = 256, kernel_size = 3, stride = 1, padding = 1)
        )
        self.pool3 = nn.Sequential(
            nn.BatchNorm2d(256),
            nn.ReLU(inplace = False),
            nn.MaxPool2d(kernel_size = 2, stride = 2)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels = 256, out_channels = 512, kernel_size = 3, stride = 1, padding = 1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace = True),
            nn.Conv2d(in_channels = 512, out_channels = 512, kernel_size = 3, stride = 1, padding = 1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace = True),
            nn.Conv2d(in_channels = 512, out_channels = 512, kernel_size = 3, stride = 1, padding = 1)
        )
        self.pool4 = nn.Sequential(
            nn.BatchNorm2d(512),
            nn.ReLU(inplace = False),
            nn.MaxPool2d(kernel_size = 2, stride = 2)
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(in_channels = 512, out_channels = 512, kernel_size = 3, stride = 1, padding = 1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace = True),
            nn.Conv2d(in_channels = 512, out_channels = 512, kernel_size = 3, stride = 1, padding = 1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace = True),
            nn.Conv2d(in_channels = 512, out_channels = 512, kernel_size = 3, stride = 1, padding = 1)
        )
        self.pool5 = nn.Sequential(
            nn.BatchNorm2d(512),
            nn.ReLU(inplace = False),
            nn.AdaptiveAvgPool2d((7, 7))
        )
        # classification part
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(inplace = True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace = True),
            nn.Dropout(),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):                                   # shape: [B, 1, 224, 224]
        conv1 = self.conv1(x)                               # shape: [B, 64, 224, 224]
        pool1 = self.pool1(conv1)                           # shape: [B, 64, 112, 112]
        conv2 = self.conv2(pool1)                           # shape: [B, 128, 112, 112]
        pool2 = self.pool2(conv2)                           # shape: [B, 128, 56, 56]
        conv3 = self.conv3(pool2)                           # shape: [B, 256, 56, 56]
        pool3 = self.pool3(conv3)                           # shape: [B, 256, 28, 28]
        conv4 = self.conv4(pool3)                           # shape: [B, 512, 28, 28]
        pool4 = self.pool4(conv4)                           # shape: [B, 512, 14, 14]
        conv5 = self.conv5(pool4)                           # shape: [B, 512, 14, 14]
        pool5 = self.pool5(conv5)                           # shape: [B, 512, 7, 7]
        pool5 = pool5.view(x.size(0), -1)                   # shape: [B, 512 * 7 * 7]
        x = self.classifier(pool5)                          # shape: [B, 1000]
        return x
        