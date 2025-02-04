from python import Python, PythonObject
from collections import Set
from collections import Dict
from structs.Nodes import NodeType, NormalizationType, PoolingType, ActivationType
from structs.Nodes import Node, InputNode, OutputNode, NormalizationNode, PoolingNode, ActivationNode, FlattenNode
from structs.Nodes import ConvolutionalNode, LinearNode, NodeTrait
from structs.SharedConv2d import SharedConv2d


struct Graph:
    var test: String
    var ALLOWED_KERNEL_SIZES: Set[Int]
    var ALLOWED_NUMBER_OF_CONVOLUTION_CHANNELS: Set[Int]
    var MAX_NUMBER_OF_CONVOLUTION_CHANNELS: Int
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
        self.MAX_NUMBER_OF_CONVOLUTION_CHANNELS = 0
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
        
        for c in self.ALLOWED_NUMBER_OF_CONVOLUTION_CHANNELS:
            if c[] > self.MAX_NUMBER_OF_CONVOLUTION_CHANNELS:
                self.MAX_NUMBER_OF_CONVOLUTION_CHANNELS = c[]
    
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
        
    def addActivationLayer(inout self): 
        """
        Adds a layer of ActivationNodes to the graph.
                  
        Returns:
            outputShape (PythonObject): Resulting tensor shape after passing 
                                        through layer as PyTorch Tensor.Size() object
        """
        uuid = Python.import_module("uuid")
        var actOptsLen: Int = len(self.activationOptions)
        for i in range(actOptsLen):
            var opt = self.activationOptions[i]
            pytorchLayerId = uuid.uuid4()
            var nodeName = 'L' + str(self.layer) + '_' + opt.__str__()
            node = Node(ActivationNode(name=nodeName, activationType=opt))
            self.addNode(node)
        self.prevNodes = self.curNodes
        self.curNodes = List[String]()
        
    def addConvolutionalLayer(inout self, layer: Int, inputShape: PythonObject) -> PythonObject:
        """
        Adds a layer of ConvolutionalNodes to the graph.
        
        Args:
            layer (Int): The current layer in the graph
            inputShape (PythonObject): The shape of the input tensor as a PyTorch Tensor.Size() object
            
        Returns:
            outputShape (PythonObject): The shape of the input tensor as a PyTorch Tensor.Size() object
        """
        uuid = Python.import_module("uuid")
        torch = Python.import_module("torch")
        var maxOutShape = inputShape
        var outShapes: List[PythonObject] = List[PythonObject](inputShape)
        for kernel in self.ALLOWED_KERNEL_SIZES:
            var pytorchLayerId = uuid.uuid4()
            for oc in self.ALLOWED_NUMBER_OF_CONVOLUTION_CHANNELS:
                var nodeName = 'L' + str(self.layer) + '_' + str(kernel[]) + 'x' + str(kernel[]) + '_Conv(oc=' + str(oc[]) + ')'
                var node = Node(ConvolutionalNode(name=nodeName, 
                                                  kernel_size=kernel[], 
                                                  maxNumInputChannels=inputShape[1], 
                                                  maxNumOutputChannels=self.MAX_NUMBER_OF_CONVOLUTION_CHANNELS, 
                                                  numOutputChannels=oc[],
                                                  layer=layer, 
                                                  pytorchLayerId=pytorchLayerId))
                self.addNode(node)
                var outShape = SharedConv2d.calcOutSize(inputShape, 8, 3)
                outShapes.append(outShape)
                
        self.prevNodes = self.curNodes
        self.curNodes = List[String]()        
        
        var maxElements = 0
        var shapesLen = len(outShapes)
        for i in range(shapesLen):
            shape = outShapes[i]
            # Compute the number of elements for this shape
            var numElements = torch.tensor(shape).prod().item()      
            if numElements > maxElements:
                maxElements = numElements
                maxOutShape = shape
        
        return maxOutShape
        
    def addConvolutionalLayers(inout self, inputShape: PythonObject) -> PythonObject:
        """
        Adds layers of ConvolutionalNodes to the graph.
        
        Args:
            inputShape (PythonObject): The shape of the input tensor as a PyTorch Tensor.Size() object
            
        Returns:
            outputShape (PythonObject): The shape of the input tensor as a PyTorch Tensor.Size() object
        """
        var outShape = inputShape
        var outShapes = List[PythonObject]()
        for i in range(self.numConvLayers):       
            self.layer = self.layer + 1
            var outShape = self.addConvolutionalLayer(self.layer, inputShape)
            self.addNormalizationLayer(outShape[1])                
            self.addActivationLayer()
            self.addPoolingLayer()
            outShapes.append(outShape)
            inputShape = outShape
        return outShape
        
    def addInputLayer(inout self, inputShape: PythonObject) -> PythonObject:
        """
        Adds a layer of InputNodes to the graph.
        
        Args:
            inputShape (PythonObject): The shape of the input tensor as a PyTorch Tensor.Size() object
            
        Returns:
            inputShape (PythonObject): The shape of the input tensor as a PyTorch Tensor.Size() object
        """
        node = Node(theNode=InputNode(inputShape=inputShape))
        self.addNode(node)
        self.prevNodes = self.curNodes
        self.curNodes = List[String]()
        return inputShape
            
    def addNormalizationLayer(inout self, numFeatures: Int): 
        """
        Adds a a layer of NormalizationNodes to the graph.
        
        Args:
            numFeatures (Int): The number of input features for the layer
        """
        uuid = Python.import_module("uuid")
        var normOptsLen: Int = len(self.normalizationOptions)
        for i in range(normOptsLen):
            var opt = self.normalizationOptions[i]
            pytorchLayerId = uuid.uuid4()
            var nodeName = 'L' + str(self.layer) + '_' + opt.__str__()
            node = Node(theNode=NormalizationNode(name=nodeName, 
                         normalizationType=opt, 
                         numFeatures=numFeatures, pytorchLayerId=pytorchLayerId))
            self.addNode(node)
        self.prevNodes = self.curNodes
        self.curNodes = List[String]()
        
    def addPoolingLayer(inout self): 
        """
        Adds a layer of PoolingNodes to the graph.
                  
        Returns:
            outputShape (PythonObject): Resulting tensor shape after passing 
                                        through layer as PyTorch Tensor.Size() object
        """
        uuid = Python.import_module("uuid")
        var poolOptsLen: Int = len(self.poolingOptions)
        for i in range(poolOptsLen):
            var opt = self.poolingOptions[i]
            pytorchLayerId = uuid.uuid4()
            var nodeName = 'L' + str(self.layer) + '_' + opt.__str__()
            node = Node(theNode=PoolingNode(name=nodeName, poolingType=opt))
            self.addNode(node)
        self.prevNodes = self.curNodes
        self.curNodes = List[String]()
        