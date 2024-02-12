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
        self.stride = (1, 1)
        self.padding = (0, 0)
        self.dilation = (1, 1)

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


    def calcOutSize(self, inputHeight, inputWidth):
        """
        Calculate the output height and width of a Conv2d layer.
        
        Parameters:
        - input_height: Height of the input tensor.
        - input_width: Width of the input tensor.
        
        Returns:
        - A tuple containing the output height and width.
        """
        # Calculate output height and width
        outputHeight = ((inputHeight + 2 * self.padding[0] - self.dilation[0] 
            * (self.kernelSize[0] - 1) - 1) // self.stride[0]) + 1
        outputWidth = ((inputWidth + 2 * self.padding[1] - self.dilation[1] 
            * (self.kernelSize[1] - 1) - 1) // self.stride[1]) + 1

        return (outputHeight, outputWidth)

