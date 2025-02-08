from collections import Optional
from python import Python, PythonObject
from memory import UnsafePointer


@value
struct SharedConv2d():
    var kernel_size: Int
    var maxInChannels: Int
    var maxOutChannels: Int
    var stride: Int
    var padding: Int
    var dilation: Int
    var weight: PythonObject
    var bias: PythonObject
    var weightSub: PythonObject
    var biasSub: PythonObject
    var device: PythonObject
    var inChannels: Int
    var outChannels: Int
    var F: PythonObject

    def __init__(inout self, in_channels: Int, out_channels: Int, kernel_size: Int):
        torch = Python.import_module("torch")
        nn = Python.import_module("torch.nn")
        init = Python.import_module("torch.nn.init")
        self.F = Python.import_module("torch.nn.functional")
        math = Python.import_module("math")
    
        self.kernel_size = kernel_size
        self.maxInChannels = in_channels
        self.maxOutChannels = out_channels
        self.device = torch.device("cpu")
        self.inChannels = self.maxInChannels
        self.outChannels = self.maxOutChannels

        # Initialize the weights and biases for the maximum configuration
        self.weight = nn.Parameter(torch.Tensor(self.maxOutChannels, self.maxInChannels)).to(self.device)
        self.bias = nn.Parameter(torch.Tensor(self.maxOutChannels)).to(self.device)
        self.weightSub = self.weight.narrow(0, 0, self.outChannels).narrow(1, 0, self.inChannels).to(self.device)
        self.biasSub = self.bias.narrow(0, 0, self.outChannels).to(self.device)
        self.stride = 1
        self.padding = 0
        self.dilation = 1

        # Initialize the weights and biases
        self.weight = nn.Parameter(torch.Tensor(self.maxOutChannels, 
                                                self.maxInChannels, 
                                                self.kernel_size, 
                                                self.kernel_size))
        self.bias = nn.Parameter(torch.Tensor(self.maxOutChannels))

        # Initialize weights using Kaiming (He) initialization
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))  
        # 'a' is the slope of the rectifier used after this layer
        # Initialize bias with a value between -1/sqrt(fan_in) and 1/sqrt(fan_in)
        var fan_in: PythonObject
        fan_in = init._calculate_fan_in_and_fan_out(self.weight)[0]
        bound = 1.0 / math.sqrt(fan_in)
        init.uniform_(self.bias, -bound, bound)
        
    fn __str__(inout self) -> String:
        var strRep = "SharedConv2d(" + str(self.maxInChannels) + ", " + str(self.maxOutChannels) + ", kernel_size=" + str(self.kernel_size) + ")"
        return strRep

    def forward(inout self, x: UnsafePointer[PythonObject]) -> PythonObject:
        var rslt = self.F.conv2d(x[], self.weightSub, self.biasSub)
        return rslt
        
    def getOutSize(self, tensorShape: PythonObject, outChannels: Optional[Int] = Optional[Int](None)) -> PythonObject:
        """
        Returns the output height and width of a Conv2d layer.
        
        Args:
            tensorShape(): Shape of tensor as torch.Size
            outChannels(Int): The number of output channels
        
        Returns:
            outputShape(PythonObject): torch.Size object containing the shape of the output tensor
        """
        # Calculate output height and width
        var inputHeight = tensorShape[2]
        var inputWidth = tensorShape[3]
        var outputChannels: Int = self.maxOutChannels
        if outChannels:
            outputChannels = outChannels.value()
        var outputShape = SharedConv2d.calcOutSize(tensorShape, outputChannels, self.kernel_size, self.stride, self.padding, self.dilation)
        return outputShape      
        
    def initSubWeights(inout self, x: PythonObject, inChannels: Int, outChannels: Int):
        self.inChannels = inChannels
        self.outChannels = outChannels
        # Dynamically select the subset of weights and biases
        self.weightSub = self.weight.narrow(0, 0, self.outChannels).narrow(1, 0, self.inChannels).to(self.device)
        self.biasSub = self.bias.narrow(0, 0, self.outChannels).to(self.device)
        
    def to(inout self, device: PythonObject):
        """
        Moves struct to do computations on the devices passed in.
        
        Args:
            device (PythonObject): PythonObject of the device to do computations on.
        """
        self.device = device
        self.weight.to(self.device)
        self.bias.to(self.device)
        self.weightSub.to(self.device)
        self.biasSub.to(self.device)        

    @staticmethod
    def calcOutSize(tensorShape: PythonObject, outChannels: Int, kernel_size: Int,  stride: Int = 1, padding: Int = 0, dilation: Int = 1) -> PythonObject:
        """
        Calculate the output height and width of a Conv2d layer.
        
        Args:
            tensorShape(PythonObject): Shape of tensor as torch.Size
            outChannels(Int): The number of output channels
            kernel_size(Int): The kernel size for each dimension
            stride(Int): The stride
            padding(Int): Size of the padding for each dimension
            dilation(Int): The size for the dilation
        
        Returns:
            outputShape(PythonObject): torch.Size object containing the shape of the output tensor
        """
        torch = Python.import_module("torch")
        # Calculate output height and width
        var inputHeight = tensorShape[2]
        var inputWidth = tensorShape[3]
        var outputHeight = ((inputHeight + 2 * padding - dilation * (kernel_size - 1) - 1) // stride) + 1
        var outputWidth = ((inputWidth + 2 * padding - dilation * (kernel_size - 1) - 1) // stride) + 1
        var outputShape = torch.Size([tensorShape[0], outChannels, outputHeight, outputWidth])
        return outputShape