from python import Python, PythonObject
from collections import Set
from collections import Dict
from structs.Nodes import NodeType, NormalizationType, PoolingType, ActivationType
from structs.Nodes import Node, InputNode, OutputNode, NormalizationNode, PoolingNode, ActivationNode, FlattenNode
from structs.Nodes import ConvolutionalNode, LinearNode, NodeTrait


struct Graph:
    var test: String
    var ALLOWED_KERNEL_SIZES: Set[Int]
    var ALLOWED_NUMBER_OF_CONVOLUTION_CHANNELS: Set[Int]
    var ALLOWED_NUMBER_OF_LINEAR_FEATURES: Set[Int]
    var nodes: Dict[String, Node]
    var edges: Dict[String, List[String]]
    var normalizationOptions: List[NormalizationType]
    var poolingOptions: List[PoolingType]
    var activationOptions: List[ActivationType]
    var numConvLayers: Int
    var numLinearLayers: Int
    var numClasses: Int
    var prevNodes: List[String]
    var curNodes: List[String]
    var layer: Int
    var sample: List[Node]
    
    fn __init__(inout self):
        self.test = 'test'
        self.ALLOWED_KERNEL_SIZES = Set[Int](3,5)
        self.ALLOWED_NUMBER_OF_CONVOLUTION_CHANNELS = Set[Int](4, 8, 16, 32)
        self.ALLOWED_NUMBER_OF_LINEAR_FEATURES = Set[Int](16, 32, 64, 128, 256)
        self.nodes = Dict[String, Node]()
        self.edges = Dict[String, List[String]]()
        self.normalizationOptions = List[NormalizationType](NormalizationType.NO_NORM, NormalizationType.BATCH_NORM)
        self.poolingOptions = List[PoolingType](PoolingType.NO_POOLING, PoolingType.MAX_POOLING)
        self.activationOptions = List[ActivationType](ActivationType.LINEAR, ActivationType.RELU)
        self.numConvLayers = 2
        self.numLinearLayers = 3
        self.numClasses = 10
        self.prevNodes = List[String]()
        self.curNodes = List[String]()
        self.layer = 0
        self.sample = List[Node]()
        
    def addInputLayer(inout self, inputShape: PythonObject):
        """
        Adds an InputNode to the graph.
        
        Args:
            inputShape (PythonObject): The shape of the input tensor as a PyTorch Tensor.Size() object
            
        Returns:
            inputShape (PythonObject): The shape of the input tensor as a PyTorch Tensor.Size() object
        """
        node = Node(theNode=InputNode(inputShape=inputShape))
        self.addNode(node)
        self.prevNodes = self.curNodes
        self.curNodes = List[String]()
        self.addNormalizationLayer(inputShape[1])
        
    def addNode(inout self, node: Node):
        """
        Adds a node to the Graph.
        
        Args:
            node (Node): The node to add to the graph
        """
        var nodeName = node.getName() 
        self.nodes[nodeName] = node
        self.edges[nodeName] = List[String]()
        prevNodesLength = len(self.prevNodes)
        for i in range(prevNodesLength):
            var prev: String = self.prevNodes[i]
            self.edges[prev].append(nodeName)
        self.curNodes.append(nodeName)
            
    def addNormalizationLayer(inout self, numFeatures: Int): 
        """
        Adds an NormalizationNode to the graph.
        
        Args:
            numFeatures (Int): The number of input features for the layer
        """
        uuid = Python.import_module("uuid")
        var normOptsLen: Int = len(self.normalizationOptions)
        for i in range(normOptsLen):
            var opt = self.normalizationOptions[i]
            pytorchLayerId = uuid.uuid4()
            var nodeName = 'L' + str(0) + '_' + opt.__str__()
            node = Node(theNode=NormalizationNode(name=nodeName, 
                         normalizationType=opt, 
                         numFeatures=numFeatures, pytorchLayerId=pytorchLayerId))
            self.addNode(node)
        self.prevNodes = self.curNodes
        self.curNodes = List[String]()
        

        