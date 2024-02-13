import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import math

class SharedConv2D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, 
                 padding=0, dilation=1, groups=1, bias=True, 
                 padding_mode='zeros', device=None, dtype=None):
        super(SharedConv2D, self).__init__()
        self.initKernelSize(kernel_size)
        self.maxInChannels = in_channels
        self.maxOutChannels = out_channels
        self.initStride(stride)
        self.initPadding(padding)
        self.initDilation(dilation)

        # Initialize the weights and biases
        self.weight = nn.Parameter(torch.Tensor(self.maxOutChannels, 
                                                self.maxInChannels, 
                                                self.kernelSize[0], 
                                                self.kernelSize[1]))
        self.bias = nn.Parameter(torch.Tensor(self.maxOutChannels))

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
        return x


    def calcOutSize(self, inputHeight, inputWidth):
        """
        Calculate the output height and width of a Conv2d layer.
        
        :param inputHeight: Height of the input tensor.
        :param inputWidth: Width of the input tensor.
        
        Returns:
        - A tuple containing the output height and width.
        """
        # Calculate output height and width
        outputHeight = ((inputHeight + 2 * self.padding[0] - self.dilation[0] 
            * (self.kernelSize[0] - 1) - 1) // self.stride[0]) + 1
        outputWidth = ((inputWidth + 2 * self.padding[1] - self.dilation[1] 
            * (self.kernelSize[1] - 1) - 1) // self.stride[1]) + 1

        return (outputHeight, outputWidth)


    def initDilation(self, dilation):
        """
        Inits and validates the dilation for the class
        
        :param self: Instance of the SharedConv2d class
        :param dilation: The passed in dilation to init
        """
        typeErrMsg = "Dilation must be either an integer, a tuple of "
        typeErrMsg += "integers, or a list 2 integers"
        if isinstance(dilation, int):
            self.dilation = (dilation, dilation)
            return
        elif isinstance(dilation, tuple):
            if (len(dilation) == 2 and isinstance(dilation[0], int)  
            and isinstance(dilation[1], int)):
                self.dilation = (dilation[0], dilation[1])
                return
        raise TypeError(typeErrMsg) 


    def initKernelSize(self, kernelSize):
        """
        Inits and validates the kernel size for the class
        
        :param self: Instance of the SharedConv2d class
        :param kernelSize: The passed in kernel size to init
        """
        typeErrMsg = "Kernel Size must be either an integer, a tuple of "
        typeErrMsg += "integers, or a list 2 integers"
        if isinstance(kernelSize, int):
            # Kernel size as int => kernel_size x kernel_size
            self.kernelSize = (kernelSize, kernelSize)
            return
        elif isinstance(kernelSize, tuple):
            if (len(kernelSize) == 2 and isinstance(kernelSize[0], int)  
            and isinstance(kernelSize[1], int)):
                self.kernelSize = (kernelSize[0], kernelSize[1])
                return
        raise TypeError(typeErrMsg)


    def initPadding(self, padding):
        """
        Inits and validates the padding for the class
        
        :param self: Instance of the SharedConv2d class
        :param padding: The passed in padding to init
        """
        typeErrMsg = "Padding must be either an integer, a tuple of "
        typeErrMsg += "integers, or a list 2 integers"
        if isinstance(padding, int):
            self.padding = (padding, padding)
            return
        elif isinstance(padding, tuple):
            if (len(padding) == 2 and isinstance(padding[0], int)  
            and isinstance(padding[1], int)):
                self.padding = (padding[0], padding[1])
                return
        raise TypeError(typeErrMsg) 


    def initStride(self, stride):
        """
        Inits and validates the stride for the class
        
        :param self: Instance of the SharedConv2d class
        :param stride: The passed in stride to init
        """
        typeErrMsg = "Stride must be either an integer, a tuple of "
        typeErrMsg += "integers, or a list 2 integers"
        if isinstance(stride, int):
            # Stride as int => stride x stride
            self.stride = (stride, stride)
            return
        elif isinstance(stride, tuple):
            if (len(stride) == 2 and isinstance(stride[0], int)  
            and isinstance(stride[1], int)):
                self.stride = (stride[0], stride[1])
                return
        raise TypeError(typeErrMsg) 
   

