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
        self.weight = nn.Parameter(torch.Tensor(self.maxOutFeatures, self.maxInFeatures))
        self.bias = nn.Parameter(torch.Tensor(self.maxOutFeatures))
        self.weightSub = self.weight.narrow(0, 0, self.outChannels).narrow(1, 0, self.inChannels).to(self.device)
        self.biasSub = self.bias.narrow(0, 0, self.outChannels).to(self.device)

        # Initialize weights using Kaiming (He) initialization
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            var bound = 1.0 / math.sqrt(self.maxInFeatures)
            init.uniform_(self.bias, -bound, bound)
        
    fn __str__(inout self) -> String:
        strRep = "SharedLinear(max_in_features=" + str(self.maxInFeatures) 
        strRep += ", max_out_features=" + str(self.maxOutFeatures) + ")"
        return strRep
        
    def initSubWeights(inout self, x: PythonObject, inChannels: Int, outChannels: Int):
        self.inChannels = inChannels
        self.outChannels = outChannels
        # Dynamically select the subset of weights and biases
        self.weightSub = self.weight.narrow(0, 0, self.outChannels).narrow(1, 0, self.inChannels).to(self.device)
        self.biasSub = self.bias.narrow(0, 0, self.outChannels).to(self.device)

    fn forward(inout self, x: PythonObject) raises -> PythonObject:
        var out = self.F.linear(x, self.weightSub, self.biasSub)   
        return out
        
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