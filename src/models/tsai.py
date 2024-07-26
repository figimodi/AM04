import torch
from torch.autograd import Variable


class TSAINetwork(torch.nn.Module):
    def __init__(self):
        super(TSAINetwork, self).__init__()
        self.conv1 = torch.nn.Conv2d(2, 64, kernel_size=4, padding=1, stride=2)
        self.conv2 = torch.nn.Conv2d(64, 64, kernel_size=4, padding=1, stride=2)
        self.conv3 = torch.nn.Conv2d(64, 128, kernel_size=4, padding=1, stride=2)
        self.conv4 = torch.nn.Conv2d(128, 128, kernel_size=4, padding=1, stride=2)
        self.conv5 = torch.nn.Conv2d(128, 256, kernel_size=4, padding=1, stride=2)
        self.conv6 = torch.nn.Conv2d(256, 256, kernel_size=4, padding=1, stride=2)
        self.conv7 = torch.nn.Conv2d(256, 512, kernel_size=4, padding=1, stride=2)
        
        self.conv8 = torch.nn.Conv2d(512, 1024, kernel_size=4, padding=0, stride=2)
        self.dec8 = torch.nn.ConvTranspose2d(1024, 512, kernel_size=4, padding=0, stride=2)
        
        self.dec7 = torch.nn.ConvTranspose2d(512, 256, kernel_size=4, padding=1, stride=2)
        self.dec6 = torch.nn.ConvTranspose2d(256, 256, kernel_size=4, padding=1, stride=2)
        self.dec5 = torch.nn.ConvTranspose2d(256, 128, kernel_size=4, padding=1, stride=2)
        self.dec4 = torch.nn.ConvTranspose2d(128, 128, kernel_size=4, padding=1, stride=2)
        self.dec3 = torch.nn.ConvTranspose2d(128, 64, kernel_size=4, padding=1, stride=2)
        self.dec2 = torch.nn.ConvTranspose2d(64, 64, kernel_size=4, padding=1, stride=2)
        self.dec1 = torch.nn.ConvTranspose2d(64, 1, kernel_size=4, padding=1, stride=2)

    def forward(self, x):
        # Y = Variable(x, requires_grad=True)
        y_conv1 = self.conv1(x)
        y_conv2 = self.conv2(y_conv1)
        y_conv3 = self.conv3(y_conv2)
        y_conv4 = self.conv4(y_conv3)
        y_conv5 = self.conv5(y_conv4)
        y_conv6 = self.conv6(y_conv5)
        y_conv7 = self.conv7(y_conv6)
        
        y_conv8 = self.conv8(y_conv7)
        y_dec8 = self.dec8(y_conv8)
        
        y_skip7 = y_conv7+y_dec8
        y_dec7 = self.dec7(y_skip7)
        
        y_skip6 = y_conv6+y_dec7
        y_dec6 = self.dec6(y_skip6)
        
        y_skip5 = y_conv5+y_dec6
        y_dec5 = self.dec5(y_skip5)
        
        y_skip4 = y_conv4+y_dec5
        y_dec4 = self.dec4(y_skip4)
        
        y_skip3 = y_conv3+y_dec4
        y_dec3 = self.dec3(y_skip3)
        
        y_skip2 = y_conv2+y_dec3
        y_dec2 = self.dec2(y_skip2)
        
        y_skip1 = y_conv1+y_dec2
        Y = self.dec1(y_skip1)
        
        return Y