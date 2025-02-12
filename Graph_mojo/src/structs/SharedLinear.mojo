from python import Python, PythonObject


@value
struct SharedLinear:
    var maxInFeatures: Int
    var maxOutFeatures: Int
    var weight: PythonObject
    var bias: PythonObject
    var weightSub: PythonObject
    var biasSub: PythonObject
    var device: PythonObject
    var inChannels: Int
    var outChannels: Int
    var F: PythonObject

    fn __init__(inout self, max_in_features: Int, max_out_features: Int) raises:
        torch = Python.import_module("torch")
        nn = Python.import_module("torch.nn")
        init = Python.import_module("torch.nn.init")
        self.F = Python.import_module("torch.nn.functional")
        math = Python.import_module("math")
        
        self.maxInFeatures = max_in_features
        self.maxOutFeatures = max_out_features
        self.device = torch.device("cpu")
        self.inChannels = self.maxInFeatures
        self.outChannels = self.maxOutFeatures

        # Initialize the weights and biases for the maximum configuration
        self.weight = torch.Tensor(self.maxOutFeatures, self.maxInFeatures).to(self.device)
        self.bias = torch.Tensor(self.maxOutFeatures).to(self.device)
        self.weight = self.weight.pin_memory()
        self.bias = self.bias.pin_memory()
        self.weightSub = nn.Parameter(self.weight.narrow(0, 0, self.outChannels).narrow(1, 0, self.inChannels))
        self.biasSub = nn.Parameter(self.bias.narrow(0, 0, self.outChannels))

        # Initialize weights using Kaiming (He) initialization
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            var bound = 1.0 / math.sqrt(self.maxInFeatures)
            init.uniform_(self.bias, -bound, bound)
        
    fn __str__(inout self) -> String:
        strRep = "SharedLinear(max_in_features=" + str(self.maxInFeatures) 
        strRep += ", max_out_features=" + str(self.maxOutFeatures) + ")"
        return strRep
        
    fn initSubWeights(inout self, inChannels: Int, outChannels: Int) raises:
        torch = Python.import_module("torch")
        nn = Python.import_module("torch.nn")
        self.inChannels = inChannels
        self.outChannels = outChannels
        # Dynamically select the subset of weights and biases
        self.weightSub = nn.Parameter(self.weight.narrow(0, 0, self.outChannels).narrow(1, 0, self.inChannels))
        self.biasSub = nn.Parameter(self.bias.narrow(0, 0, self.outChannels))

    fn forward(inout self, x: PythonObject) raises -> PythonObject:
        return self.F.linear(x, self.weightSub, self.biasSub)
        
    def to(inout self, device: PythonObject):
        """
        Moves struct to do computations on the devices passed in.
        
        Args:
            device (PythonObject): PythonObject of the device to do computations on.
        """
        nn = Python.import_module("torch.nn")
        self.device = device
        self.weight = self.weight.to(self.device)
        self.bias = self.bias.to(self.device)
        self.weightSub = nn.Parameter(self.weight.narrow(0, 0, self.outChannels).narrow(1, 0, self.inChannels))
        self.biasSub = nn.Parameter(self.bias.narrow(0, 0, self.outChannels))