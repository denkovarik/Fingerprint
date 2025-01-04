from python import Python, PythonObject


struct SharedLinear:
    var maxInFeatures: Int
    var maxOutFeatures: Int
    var weight: PythonObject
    var bias: PythonObject

    fn __init__(inout self, max_in_features: Int, max_out_features: Int) raises:
        torch = Python.import_module("torch")
        nn = Python.import_module("torch.nn")
        init = Python.import_module("torch.nn.init")
        F = Python.import_module("torch.nn.functional")
        math = Python.import_module("math")
        
        self.maxInFeatures = max_in_features
        self.maxOutFeatures = max_out_features

        # Initialize the weights and biases for the maximum configuration
        self.weight = nn.Parameter(torch.Tensor(self.maxOutFeatures, self.maxInFeatures))
        self.bias = nn.Parameter(torch.Tensor(self.maxOutFeatures))

        # Initialize weights using Kaiming (He) initialization
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            var bound = 1.0 / math.sqrt(self.maxInFeatures)
            init.uniform_(self.bias, -bound, bound)
        
    fn __str__(inout self) -> String:
        strRep = "SharedLinear(max_in_features=" + str(self.maxInFeatures) 
        strRep += ", max_out_features=" + str(self.maxOutFeatures) + ")"
        return strRep

    def forward(inout self, x: PythonObject, inChannels: Int, outChannels: Int) -> PythonObject:
        F = Python.import_module("torch.nn.functional")
        # Dynamically select the subset of weights and biases
        var weight = self.weight.narrow(0, 0, outChannels)  # slices on the 0th dimension (out_channels)
        weight = weight.narrow(1, 0, inChannels)  # slices on the 1st dimension (in_channels)
        var bias = self.bias.narrow(0, 0, outChannels)  # slices on the 0th dimension (out_channels)
        var out = F.linear(x, weight, bias)   
        return out
        