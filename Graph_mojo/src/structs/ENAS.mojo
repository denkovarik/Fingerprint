from python import Python, PythonObject
from structs.Nodes import NodeType, NormalizationType, PoolingType, ActivationType
from structs.Nodes import Node, InputNode, OutputNode, NormalizationNode, PoolingNode, ActivationNode, FlattenNode
from structs.Nodes import ConvolutionalNode, LinearNode, NodeTrait
from structs.Graph import GraphHandler, Graph
from memory import UnsafePointer
from collections import Dict


@value
struct CustomCNN:
    var graph: Graph
    var layers: List[UnsafePointer[Node]]
    var torch: PythonObject
    var nn: PythonObject
    var F: PythonObject
    var parameters: PythonObject
    var inputShape: PythonObject
    var device: PythonObject

    fn __init__(inout self, graph: Graph, inputShape: PythonObject)raises:
        self.torch = Python.import_module("torch")
        self.nn = Python.import_module("torch.nn")
        self.F = Python.import_module("torch.nn.functional")
        self.graph = graph
        self.layers = List[UnsafePointer[Node]]()
        self.parameters = Python.list()
        self.inputShape = inputShape
        self.device = self.torch.device("cpu")
        
        var nodeName: String = 'input'        
        if graph.nodes.__contains__(nodeName) == True:
            var ind: Int = 0
            self.layers.append(graph.nodes[nodeName])
            while nodeName != 'output':
                nodeName = graph.edges[nodeName][0]
                self.layers.append(graph.nodes[nodeName])    
            self.registerParameters()
            
    fn initSubweights(inout self) raises:
        var randTense: PythonObject = self.torch.rand(self.inputShape).to(self.device)
        
        for i in range(self.layers.__len__()):
            var inChannels: Int = randTense.shape[1]
            print(self.layers[i][].getName())
            self.layers[i][].initSubWeights(inChannels)
            randTense = self.layers[i][].forward(randTense)            
            
    def to(inout self, device: PythonObject):
        self.device = device
        for i in range(self.layers.__len__()):
            self.layers[i][].to(device)

    def forward(self, x: PythonObject) -> PythonObject:
        var out: PythonObject = x
        for i in range(self.layers.__len__()):
            out = self.layers[i][].forward(out)    
        return out
        
    def registerParameters(inout self) -> PythonObject:
        self.parameters = Python.list()
        for i in range(self.layers.__len__()):        
            if self.layers[i][].nodeType.value == NodeType.CONVOLUTION.value:
                pass
                self.parameters.append(self.layers[i][].node[ConvolutionalNode].pytorchLayer.weightSub)
                self.parameters.append(self.layers[i][].node[ConvolutionalNode].pytorchLayer.biasSub)
            elif self.layers[i][].nodeType.value == NodeType.NORMALIZATION.value:
                pass
                self.parameters.append(self.layers[i][].node[NormalizationNode].pytorchLayer.weight)
                self.parameters.append(self.layers[i][].node[NormalizationNode].pytorchLayer.bias)
            elif self.layers[i][].nodeType.value == NodeType.LINEAR.value:
                pass
                self.parameters.append(self.layers[i][].node[LinearNode].pytorchLayer.weightSub)
                self.parameters.append(self.layers[i][].node[LinearNode].pytorchLayer.biasSub)
        return self.parameters

    fn zero_grad(inout self):
        for param in self.parameters.values():
            try:
                param.grad.zero_()
            except AttributeError:
                pass


struct ENAS:
    var graphHandler: GraphHandler
    var inputShape: PythonObject
    var sampleGraph: Graph
    var sample: CustomCNN
    
    fn __init__(inout self, inputShape: PythonObject) raises:
        torch = Python.import_module("torch")
        self.graphHandler = GraphHandler()
        self.inputShape=torch.Size([64, 3, 32, 32])
        self.sampleGraph = Graph()
        self.sample = CustomCNN(Graph(), inputShape)
        
    def construct(inout self):
        self.graphHandler.construct(inputShape=self.inputShape)
        
    def sampleArchitecture(inout self, sample: List[Int]):
        self.sampleGraph = self.graphHandler.sampleArchitecture(sample)
        self.sample = CustomCNN(self.sampleGraph, self.inputShape)



