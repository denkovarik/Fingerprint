import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import math

class SharedConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, 
                 padding=0, dilation=1, groups=1, bias=True, 
                 padding_mode='zeros', device=None, dtype=None):
        super(SharedConv2d, self).__init__()
        self.initKernelSize(kernel_size)
        self.maxInChannels = in_channels
        self.maxOutChannels = out_channels
        self.initStride(stride)
        self.initPadding(padding)
        self.initDilation(dilation)

        # Initialize the weights and biases
        self.weight = nn.Parameter(torch.Tensor(self.maxOutChannels, 
                                                self.maxInChannels, 
                                                self.kernel_size[0], 
                                                self.kernel_size[1]))
        self.bias = nn.Parameter(torch.Tensor(self.maxOutChannels))

        # Initialize weights using Kaiming (He) initialization
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))  
        # 'a' is the slope of the rectifier used after this layer
        if self.bias is not None:
            # Initialize bias with a value between -1/sqrt(fan_in) and 1/sqrt(fan_in)
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)


    def __str__(self):
        strRep = (f"SharedConv2d("
                 f"{self.maxInChannels}, {self.maxOutChannels}, "
                 f"kernel_size=({self.kernel_size[0]}, {self.kernel_size[1]})"
                 f")")
        return strRep


    def forward(self, x, inChannels, outChannels):
        weight = self.weight[:outChannels, :inChannels, :, :]
        bias = self.bias[:outChannels]
        x = F.conv2d(x, weight, bias)
        return x


    def getOutSize(self, tensorShape, out_channels=None):
        """
        Calculate the output height and width of a Conv2d layer.
        
        :param tensorShape: Shape of tensor as torch.Size
        :param out_channels: Default param for the number of Output channels
        
        Returns:
        - A torch.Size object containing the shape of the output tensor
        """
        outChannels = out_channels
        # Error Checking
        if not isinstance(tensorShape, (tuple, torch.Size)) or len(tensorShape) != 4:
            raise ValueError("Tensor Shape must be in form of (batch_size, in_channels, height, width)")

        # Calculate output height and width
        _, _, inputHeight, inputWidth = tensorShape
        if outChannels is None:
            outChannels = self.maxOutChannels
        
        outputShape = SharedConv2d.calcOutSize(tensorShape, 
                                               outChannels, 
                                               self.kernel_size,  
                                               self.stride, 
                                               self.padding, 
                                               self.dilation)

        return outputShape


    def initDilation(self, dilation):
        """
        Inits and validates the dilation for the class
        
        :param self: Instance of the SharedConv2d class
        :param dilation: The passed in dilation to init
        """
        self.dilation = SharedConv2d.checkDilation(dilation)


    def initKernelSize(self, kernel_size):
        """
        Inits and validates the kernel size for the class
        
        :param self: Instance of the SharedConv2d class
        :param kernel_size: The passed in kernel size to init
        """
        self.kernel_size = SharedConv2d.checkKernelSize(kernel_size)


    def initPadding(self, padding):
        """
        Inits and validates the padding for the class
        
        :param self: Instance of the SharedConv2d class
        :param padding: The passed in padding to init
        """
        self.padding = SharedConv2d.checkPadding(padding)


    def initStride(self, stride):
        """
        Inits and validates the stride for the class
        
        :param self: Instance of the SharedConv2d class
        :param stride: The passed in stride to init
        """
        self.stride = SharedConv2d.checkStride(stride)

    
    @staticmethod
    def calcOutSize(tensorShape, out_channels, kernel_size,  stride=1, padding=0, dilation=1):
        """
        Calculate the output height and width of a Conv2d layer.
        
        :param tensorShape: Shape of tensor as torch.Size
        :param inputWidth: Width of the input tensor.
        
        Returns:
        - A torch.Size object containing the shape of the output tensor
        """
        outChannels = out_channels
        # Error Checking
        if not isinstance(tensorShape, (tuple, torch.Size)) or len(tensorShape) != 4:
            raise ValueError("Tensor Shape must be in form of (batch_size, in_channels, height, width)")
    
        # Calculate output height and width
        _, _, inputHeight, inputWidth = tensorShape
    
        # Make sure kernel_size is tuple of ints
        kernel_size = SharedConv2d.checkKernelSize(kernel_size)
        # Make sure padding is a tuple of ints
        padding = SharedConv2d.checkPadding(padding)
        # Make sure stride is a tuple of ints
        stride = SharedConv2d.checkStride(stride)
        # Make sure dilation is a tuple of ints
        dilation = SharedConv2d.checkDilation(dilation)

        outputHeight = ((inputHeight + 2 * padding[0] - dilation[0] 
            * (kernel_size[0] - 1) - 1) // stride[0]) + 1
        outputWidth = ((inputWidth + 2 * padding[1] - dilation[1] 
            * (kernel_size[1] - 1) - 1) // stride[1]) + 1

        outputShape = torch.Size([tensorShape[0], outChannels, outputHeight, outputWidth])

        return outputShape


    @staticmethod
    def checkDilation(dilation):
        """
        Validates that the dilation passed in is either an int or a tuple of 
        ints. If dilation is just an int, it will be converted to a tuple of 
        ints. Otherwise this method will raise an exception.
        
        :param dilation: Value for the dilation
        
        Returns:
        - The dilation as a tuple of ints
        """
        if SharedConv2d.isTupleOfInts(dilation):
            return dilation
        elif isinstance(dilation, int):
            return SharedConv2d.cnvrtInt2Tuple(dilation)
        typeErrMsg = "Dilation must be either an integer, a tuple of "
        typeErrMsg += "integers, or a list 2 integers"
        raise TypeError(typeErrMsg)  


    @staticmethod
    def checkKernelSize(kernel_size):
        """
        Validates that the kernel size passed in is either an int or a tuple of 
        ints. If kernel size is just an int, it will be converted to a tuple of 
        ints. Otherwise this method will raise an exception.
        
        :param kernel_size: Value for the kernel size
        
        Returns:
        - The kernel size as a tuple of ints
        """
        if SharedConv2d.isTupleOfInts(kernel_size):
            return kernel_size
        elif isinstance(kernel_size, int):
            return SharedConv2d.cnvrtInt2Tuple(kernel_size)
        typeErrMsg = "Kernel Size must be either an integer, a tuple of "
        typeErrMsg += "integers, or a list 2 integers"
        raise TypeError(typeErrMsg)  


    @staticmethod
    def checkPadding(padding):
        """
        Validates that the padding passed in is either an int or a tuple of 
        ints. If padding is just an int, it will be converted to a tuple of 
        ints. Otherwise this method will raise an exception.
        
        :param padding: Value for the dilation
        
        Returns:
        - The padding as a tuple of ints
        """
        if SharedConv2d.isTupleOfInts(padding):
            return padding
        elif isinstance(padding, int):
            return SharedConv2d.cnvrtInt2Tuple(padding)
        typeErrMsg = "Padding must be either an integer, a tuple of "
        typeErrMsg += "integers, or a list 2 integers"
        raise TypeError(typeErrMsg)  


    @staticmethod
    def checkStride(stride):
        """
        Validates that the stride passed in is either an int or a tuple of 
        ints. If stride is just an int, it will be converted to a tuple of 
        ints. Otherwise this method will raise an exception.
        
        :param dilation: Value for the dilation
        
        Returns:
        - The dilation as a tuple of ints
        """
        if SharedConv2d.isTupleOfInts(stride):
            return stride
        elif isinstance(stride, int):
            return SharedConv2d.cnvrtInt2Tuple(stride)
        typeErrMsg = "Stride must be either an integer, a tuple of "
        typeErrMsg += "integers, or a list 2 integers"
        raise TypeError(typeErrMsg)  
        

    @staticmethod
    def cnvrtInt2Tuple(intVal):
        """
        Converts the value passed in 'intVal' into a tuple of ints.

        :param intVal: Int value  to convert to a tuple of ints
        
        Returns:
        - The intVal as a tuple of ints
        """
        if not isinstance(intVal, int):
            raise TypeError("intVal must be an int")
        return (intVal, intVal)


    @staticmethod
    def isTupleOfInts(val):
        """
        Determines if 'val' is a tuple of ints

        :param val: The variable to check the type for
        
        Returns:
        - True if 'val' is a tuple of ints
        - False otherwise
        """
        if (isinstance(val, tuple) and len(val) == 2 
        and isinstance(val[0], int) and isinstance(val[1], int)):
            return True
        return False
        
   

