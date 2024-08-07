import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F


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
class NewTSAINetwork(nn.Module):
    def __init__(self):
        super(NewTSAINetwork, self).__init__()
        
        # Encoder
        self.conv0 = nn.Conv2d(2, 64, kernel_size=4, stride=2, padding=1)
        self.bn0 = nn.BatchNorm2d(64)
        self.elu0 = nn.ELU()
        
        self.conv1 = nn.Conv2d(64, 64, kernel_size=4, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.elu1 = nn.ELU()
        
        self.conv2 = nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.elu2 = nn.ELU()
        
        self.conv3 = nn.Conv2d(128, 128, kernel_size=4, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.elu3 = nn.ELU()
        
        self.conv4 = nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        self.elu4 = nn.ELU()
        
        self.conv5 = nn.Conv2d(256, 256, kernel_size=4, stride=2, padding=1)
        self.bn5 = nn.BatchNorm2d(256)
        self.elu5 = nn.ELU()
        
        self.conv6 = nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1)
        self.bn6 = nn.BatchNorm2d(512)
        self.elu6 = nn.ELU()
        
        self.fc7 = nn.Linear(512*4*4, 1024)
        
        # Decoder
        self.deconv6 = nn.ConvTranspose2d(1024, 512, kernel_size=4, stride=2, padding=1)
        self.bn6_deconv = nn.BatchNorm2d(512)
        self.elu6_deconv = nn.ELU()
        
        self.deconv5 = nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1)
        self.bn5_deconv = nn.BatchNorm2d(256)
        self.elu5_deconv = nn.ELU()
        
        self.deconv4 = nn.ConvTranspose2d(256, 256, kernel_size=4, stride=2, padding=1)
        self.bn4_deconv = nn.BatchNorm2d(256)
        self.elu4_deconv = nn.ELU()
        
        self.deconv3 = nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1)
        self.bn3_deconv = nn.BatchNorm2d(128)
        self.elu3_deconv = nn.ELU()
        
        self.deconv2 = nn.ConvTranspose2d(128, 128, kernel_size=4, stride=2, padding=1)
        self.bn2_deconv = nn.BatchNorm2d(128)
        self.elu2_deconv = nn.ELU()
        
        self.deconv1 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)
        self.bn1_deconv = nn.BatchNorm2d(64)
        self.elu1_deconv = nn.ELU()
        
        self.deconv0 = nn.ConvTranspose2d(64, 64, kernel_size=4, stride=2, padding=1)
        self.bn0_deconv = nn.BatchNorm2d(64)
        self.elu0_deconv = nn.ELU()
        
        self.output_seg = nn.ConvTranspose2d(64, 64, kernel_size=4, stride=2, padding=1)
        self.bn_output = nn.BatchNorm2d(64)
        self.elu_output = nn.ELU()

        self.final_upsample = nn.ConvTranspose2d(64, 64, kernel_size=4, stride=2, padding=1)
        self.bn_final_upsample = nn.BatchNorm2d(64)
        self.elu_final_upsample = nn.ELU()
        
        self.output_final = nn.Conv2d(64, 1, kernel_size=1)

    def forward(self, x):
        # Encoder
        conv0 = self.elu0(self.bn0(self.conv0(x)))
        conv1 = self.elu1(self.bn1(self.conv1(conv0)))
        conv2 = self.elu2(self.bn2(self.conv2(conv1)))
        conv3 = self.elu3(self.bn3(self.conv3(conv2)))
        conv4 = self.elu4(self.bn4(self.conv4(conv3)))
        conv5 = self.elu5(self.bn5(self.conv5(conv4)))
        conv6 = self.elu6(self.bn6(self.conv6(conv5)))
        fc7 = self.fc7(conv6.view(conv6.size(0), -1)).view(-1, 1024, 1, 1)
        # Decoder
        deconv6 = self.elu6_deconv(self.bn6_deconv(self.deconv6(fc7)))
        skip6 = deconv6 + F.interpolate(conv6, size=deconv6.size()[2:], mode='nearest')
        deconv5 = self.elu5_deconv(self.bn5_deconv(self.deconv5(skip6)))
        skip5 = deconv5 + F.interpolate(conv5, size=deconv5.size()[2:], mode='nearest')
        deconv4 = self.elu4_deconv(self.bn4_deconv(self.deconv4(skip5)))
        skip4 = deconv4 + F.interpolate(conv4, size=deconv4.size()[2:], mode='nearest')
        deconv3 = self.elu3_deconv(self.bn3_deconv(self.deconv3(skip4)))
        skip3 = deconv3 + F.interpolate(conv3, size=deconv3.size()[2:], mode='nearest')
        deconv2 = self.elu2_deconv(self.bn2_deconv(self.deconv2(skip3)))
        skip2 = deconv2 + F.interpolate(conv2, size=deconv2.size()[2:], mode='nearest')
        deconv1 = self.elu1_deconv(self.bn1_deconv(self.deconv1(skip2)))
        skip1 = deconv1 + F.interpolate(conv1, size=deconv1.size()[2:], mode='nearest')
        deconv0 = self.elu0_deconv(self.bn0_deconv(self.deconv0(skip1)))
        skip0 = deconv0 + F.interpolate(conv0, size=deconv0.size()[2:], mode='nearest')
        output_seg = self.elu_output(self.bn_output(self.output_seg(skip0)))
        final_upsample = self.elu_final_upsample(self.bn_final_upsample(self.final_upsample(output_seg)))
        output_final = self.output_final(final_upsample)
        return output_final
