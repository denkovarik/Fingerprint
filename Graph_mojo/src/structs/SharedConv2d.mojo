from collections import Optional
from python import Python, PythonObject


struct SharedConv2d():
    var kernel_size: Int
    var maxInChannels: Int
    var maxOutChannels: Int
    var stride: Int
    var padding: Int
    var dilation: Int
    var weight: PythonObject
    var bias: PythonObject

    def __init__(inout self, in_channels: Int, out_channels: Int, kernel_size: Int):
        torch = Python.import_module("torch")
        nn = Python.import_module("torch.nn")
        init = Python.import_module("torch.nn.init")
        F = Python.import_module("torch.nn.functional")
        math = Python.import_module("math")
    
        self.kernel_size = kernel_size
        self.maxInChannels = in_channels
        self.maxOutChannels = out_channels
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

    def forward(inout self, x: PythonObject, inChannels: Int, outChannels: Int) -> PythonObject:
        F = Python.import_module("torch.nn.functional")
        var weight = self.weight.narrow(0, 0, outChannels).narrow(1, 0, inChannels) 
        var bias = self.bias.narrow(0, 0, outChannels)  # slices on the 0th dimension (out_channels)
        var rslt = F.conv2d(x, weight, bias)
        return rslt

        