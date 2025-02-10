from python import Python, PythonObject
from structs.Nodes import NodeType, NormalizationType, PoolingType, ActivationType
from structs.Graph import GraphHandler, Graph


@value
struct CustomCNN:
    var graph: Graph
    var parameters: Dict[String, PythonObject]
    var torch = PythonObject
    var nn = PythonObject
    var F = PythonObject
    var optim = PythonObject

    fn __init__(inout self, layers: List[PythonObject], inputShape: PythonObject, lr: Float32):
        self.torch = Python.import_module("torch")
        self.nn = Python.import_module("torch.nn")
        self.F = Python.import_module("torch.nn.functional")
        self.optim = Python.import_module("torch.optim")
        
        self.layers = layers
        self.parameters = Dict[String, PythonObject]()
        # Register parameters from custom layers
        for i in range(len(layers)):
            for j in range(len(layers[i].parameters())):
                param = layers[i].parameters()[j]
                param_name = f'param_{i}_{j}'
                self.parameters[param_name] = param

    def to(self, device):
        self.graph.to(device)

    def forward(self, x: PythonObject) -> PythonObject:
        for layer in self.layers:
            x = layer(x)
        return x

    def zero_grad(self):
        # Zero out gradients of all parameters
        for param in self.parameters.values():
            if hasattr(param, 'grad'):
                param.grad.zero_()

    def state_dict(self) -> Dict[String, PythonObject]:
        # Return a dictionary of all the parameters
        return self.parameters

    def load_state_dict(self, state_dict: Dict[String, PythonObject]):
        # Load parameters from a state dictionary
        for key, value in state_dict.items():
            self.parameters[key].copy_(value)








struct CustomCNN:
    var graph: Graph
    
    fn __init__(self, graph: Graph, inputShape: PythonObject):
        self.graph = graph
        
    def to(self, device):
        self.graph.to(device)

    fn forward(self, x) raises -> PythonObject:        
        for layer in self.graph.nodes.items():
            x = item[].value[].forward(x)
        
        return x


struct ENAS:
    var graphHandler: GraphHandler
    var inputShape: PythonObject
    var sampleGraph: Graph
    
    fn __init__(inout self) raises:
        torch = Python.import_module("torch")
        self.graphHandler = GraphHandler()
        self.inputShape=torch.Size([4, 3, 32, 32])
        self.sampleGraph = Graph()
        
    def construct(inout self):
        self.graphHandler.construct(inputShape=self.inputShape)
        
    def sampleArchitecture(inout self, sample: List[Int]):
        self.sampleGraph = self.graphHandler.sampleArchitecture(sample)



