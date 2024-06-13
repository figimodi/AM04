import torch
from torch.autograd import Variable


class TSAIEncoder(torch.nn.Module):
    def __init__(self):
        super(TSAIEncoder, self).__init__()
        self.conv1 = torch.nn.Conv2d(3, 3, kernel_size=1, padding=0)
        # TODO: Implement network

    def forward(self, x):
        Y = Variable(x, requires_grad=True)
        x = self.conv1(x)
        # TODO: Implement network
        return Y

class TSAIDecoder(torch.nn.Module):
    def __init__(self):
        super(TSAIDecoder, self).__init__()
        # TODO: Implement network

    def forward(self, x):
        # TODO: Implement network
        return x