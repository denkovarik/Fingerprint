import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import math

class SharedConv2D(nn.Module):
    def __init__(self, kernelSize, maxInChannels, maxOutChannels, useMaxPool=False):
        super(SharedConv2D, self).__init__()
        typeErrMsg = "Kernel Size must be either an integer, a tuple of "
        typeErrMsg += "integers, or a list 2 integers"

        if isinstance(kernelSize, int):
            # Kernel size as int => kernelSize x kernelSize
            self.kernelSize = (kernelSize, kernelSize)
        elif isinstance(kernelSize, list) or isinstance(kernelSize, tuple):
            if (len(kernelSize) == 2 and isinstance(kernelSize[0], int)  
            and isinstance(kernelSize[1], int)):
                self.kernelSize = (kernelSize[0], kernelSize[1])
            else:
                raise TypeError(typeErrMsg)
        else:
            raise TypeError(typeErrMsg) 
        self.maxInChannels = maxInChannels
        self.maxOutChannels = maxOutChannels
        self.useMaxPool = useMaxPool

        # Initialize the weights and biases
        self.weight = nn.Parameter(torch.Tensor(maxOutChannels, maxInChannels, 
                                   self.kernelSize[0], self.kernelSize[1]))
        self.bias = nn.Parameter(torch.Tensor(maxOutChannels))
        
        self.pool = nn.MaxPool2d(2,2)

        # Initialize weights using Kaiming (He) initialization
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))  
        # 'a' is the slope of the rectifier used after this layer
        if self.bias is not None:
            # Initialize bias with a value between -1/sqrt(fan_in) and 1/sqrt(fan_in)
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)


    def forward(self, x, inChannels, outChannels):
        weight = self.weight[:outChannels, :inChannels, :, :]
        bias = self.bias[:outChannels]
        x = F.conv2d(x, weight, bias)
        if self.useMaxPool:
            x = self.pool(x)
        return x
