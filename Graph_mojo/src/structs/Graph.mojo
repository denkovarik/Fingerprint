from python import Python, PythonObject
from collections import Set
from collections import Dict
from structs.Nodes import NodeType, NormalizationType, PoolingType, ActivationType
from structs.Nodes import Node, InputNode, OutputNode, NormalizationNode, PoolingNode, ActivationNode, FlattenNode
from structs.Nodes import ConvolutionalNode, LinearNode, NodeTrait
from structs.SharedConv2d import SharedConv2d
from memory import UnsafePointer


struct Graph:
    # Witness Me!
    var nodes: Dict[String, UnsafePointer[Node]]
    var edges: Dict[String, List[String]]
    
    fn __init__(inout self):
        self.nodes = Dict[String, UnsafePointer[Node]]()
        self.edges = Dict[String, List[String]]()
        
    fn __copyinit__(inout self, other: Self):
        self.nodes = other.nodes
        self.edges = other.edges
        
    def to(inout self, device: PythonObject):
        for item in grph.graph.nodes.items():
            item[].value[].to(device)


struct GraphHandler:
    var ALLOWED_KERNEL_SIZES: Set[Int]
    var ALLOWED_NUMBER_OF_CONVOLUTION_CHANNELS: Set[Int]
    var MAX_NUMBER_OF_CONVOLUTION_CHANNELS: Int
    var ALLOWED_NUMBER_OF_LINEAR_FEATURES: Set[Int]
    var MAX_NUMBER_OF_LINEAR_FEATURES: Int
    var graph: Graph
    var normalizationOptions: List[NormalizationType]
    var poolingOptions: List[PoolingType]
    var activationOptions: List[ActivationType]
    var numConvLayers: Int
    var numLinearLayers: Int
    var numClasses: Int
    var prevNodes: List[String]
    var curNodes: List[String]
    var layer: Int
    var sample: Graph
    var dfsStack: List[Int]
    var dfsStackKeys: List[String]
    var sampleArchitecturesEnd: Bool
    
    fn __init__(inout self):
        self.ALLOWED_KERNEL_SIZES = Set[Int](3,5)
        self.ALLOWED_NUMBER_OF_CONVOLUTION_CHANNELS = Set[Int](4, 8, 16, 32)
        self.MAX_NUMBER_OF_CONVOLUTION_CHANNELS = 0
        self.ALLOWED_NUMBER_OF_LINEAR_FEATURES = Set[Int](16, 32, 64, 128, 256)
        self.MAX_NUMBER_OF_LINEAR_FEATURES = 0
        self.graph = Graph()
        self.normalizationOptions = List[NormalizationType](NormalizationType.NO_NORM, NormalizationType.BATCH_NORM)
        self.poolingOptions = List[PoolingType](PoolingType.NO_POOLING, PoolingType.MAX_POOLING)
        self.activationOptions = List[ActivationType](ActivationType.LINEAR, ActivationType.RELU)
        self.numConvLayers = 2
        self.numLinearLayers = 3
        self.numClasses = 10
        self.prevNodes = List[String]()
        self.curNodes = List[String]()
        self.layer = 0
        self.dfsStack = List[Int]()
        self.dfsStackKeys = List[String]()
        self.sample = Graph()
        self.sampleArchitecturesEnd = True
        # Find the max number of features for convolutional layers
        for c in self.ALLOWED_NUMBER_OF_CONVOLUTION_CHANNELS:
            if c[] > self.MAX_NUMBER_OF_CONVOLUTION_CHANNELS:
                self.MAX_NUMBER_OF_CONVOLUTION_CHANNELS = c[]
        # Find the max number of features for Linear Layers        
        for c in self.ALLOWED_NUMBER_OF_LINEAR_FEATURES:
            if c[] > self.MAX_NUMBER_OF_LINEAR_FEATURES:
                self.MAX_NUMBER_OF_LINEAR_FEATURES = c[]
    
    def addNode(inout self, node: Node):
        """
        Adds a node to the Graph.
        
        Args:
            node (Node): The node to add to the graph
        """
        var nodeName = node.getName()         
        self.graph.nodes[nodeName] = UnsafePointer[Node].alloc(1)
        self.graph.nodes[nodeName].init_pointee_copy(node)
        self.graph.edges[nodeName] = List[String]()
        prevNodesLength = len(self.prevNodes)
        for i in range(prevNodesLength):
            var prev: String = self.prevNodes[i]
            self.graph.edges[prev].append(nodeName)
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
        
    def addFlattenLayer(inout self, inputShape: PythonObject) -> PythonObject:
        """
        Adds a layer of FlattenNodes to the graph.
                  
        Returns:
            outputShape (PythonObject): Resulting tensor shape after passing 
                                        through layer as PyTorch Tensor.Size() object
        """
        torch = Python.import_module("torch")
        node = Node(FlattenNode(name='L' + str(self.layer) + '_' + 'flatten'))
        self.addNode(node)
        self.prevNodes = self.curNodes
        self.curNodes = List[String]()
        
        var numElements = 1
        # Start from the second dimension (index 1)
        var numDim = len(inputShape)
        for i in range(numDim):
            # Skip first dimension
            if i > 0: 
                numElements = numElements * inputShape[i]
        
        flattenedShape = torch.tensor(numElements)
        outShape = torch.Size([inputShape[0], flattenedShape])
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
        self.addNormalizationLayer(inputShape[1])
        return inputShape
        
    fn addLinearLayer(inout self, layer: Int, inputShape: PythonObject) raises -> PythonObject: 
        """
        Adds a layer of LinearlNodes to the graph.
        
        Args:
            layer (Int): The current layer in the graph
            inputShape (PythonObject): The shape of the input tensor as a PyTorch Tensor.Size() object
            
        Returns:
            outputShape (PythonObject): The shape of the input tensor as a PyTorch Tensor.Size() object
        """
        uuid = Python.import_module("uuid")
        torch = Python.import_module("torch")
        for of in self.ALLOWED_NUMBER_OF_LINEAR_FEATURES:
            nodeName = 'L' + str(layer) + '_Linear(of=' + str(of[]) + ')'
            var pytorchLayerId = uuid.uuid4()
            node = Node(LinearNode(name=nodeName, 
                                   maxNumInFeatures=inputShape[1], 
                                   maxNumOutFeatures=self.MAX_NUMBER_OF_LINEAR_FEATURES,
                                   numOutFeatures=of[], 
                                   layer=layer, 
                                   pytorchLayerId=pytorchLayerId))
            self.addNode(node)             
        self.prevNodes = self.curNodes
        self.curNodes = List[String]()
        var outShape: PythonObject = torch.Size([inputShape[0], self.MAX_NUMBER_OF_LINEAR_FEATURES])
        return outShape
            
    def addLinearLayers(inout self, inputShape: PythonObject):
        """
        Adds layers of LinearNodes to the graph.
        
        Args:
            inputShape (PythonObject): The shape of the input tensor as a PyTorch Tensor.Size() object
        """
        uuid = Python.import_module("uuid")
        torch = Python.import_module("torch")
        var outShapes = List[PythonObject]()
        var outShape = inputShape
        for i in range(self.numLinearLayers - 1):
            var inShape: PythonObject = torch.Size([outShape[0], outShape[1]])
            self.layer += 1            
            outShape = self.addLinearLayer(self.layer, inShape)
            self.addActivationLayer()                                                  
        self.layer += 1            
        nodeName = 'L' + str(self.layer) + '_Linear(of=' + str(10) + ')'  
        # Add the final linear layer
        var pytorchLayerId = uuid.uuid4()
        node = Node(LinearNode(name=nodeName, 
                               maxNumInFeatures=outShape[1], 
                               maxNumOutFeatures=self.numClasses,
                               numOutFeatures=self.numClasses, 
                               layer=self.layer, 
                               pytorchLayerId=pytorchLayerId))
        self.addNode(node)                 
        self.prevNodes = self.curNodes
        self.curNodes = List[String]()
        self.addActivationLayer()
            
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
        
    def addOutputLayer(inout self):
        node = Node(theNode=OutputNode()) 
        self.addNode(node)
        
    def addPoolingLayer(inout self): 
        """
        Adds a layer of PoolingNodes to the graph.
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
        
    def construct(inout self, inputShape: PythonObject):
        """
        Constructs the graph.
        
        Args:
            inputShape (PythonObject): The shape of the input tensor as a PyTorch Tensor.Size() object
        """
        self.layer = 0
        self.graph = Graph()
        self.prevNodes = List[String]()
        self.curNodes = List[String]()  
        var inputOutShape = self.addInputLayer(inputShape)    
        var convOutShape= self.addConvolutionalLayers(inputOutShape)       
        var flattenOutShape = self.addFlattenLayer(convOutShape)
        var linearOutShape = self.addLinearLayers(flattenOutShape)            
        self.addOutputLayer()
        self.initDfsStack()
        self.sampleArchitecturesEnd = False
        
    def initDfsStack(inout self):
        """
        Initializes the stack used to iterate of DFS paths in Graph.
        """
        var curDfsNode: String = 'input'
        self.dfsStack = List[Int]()
        self.dfsStackKeys = List[String]()
 
        while curDfsNode != 'output':
            self.dfsStack.append(0)
            self.dfsStackKeys.append(curDfsNode)
            curDfsNode = self.graph.edges[curDfsNode][0]
        self.sampleArchitecturesEnd = False
            
    def incSampleArchitecture(inout self) -> Bool:
        var curDepth: Int = len(self.dfsStackKeys) - 1
        if self.dfsStack[curDepth] >= len(self.graph.edges[self.dfsStackKeys[curDepth]]) - 1:
            while curDepth >= 0 and self.dfsStack[curDepth] >= len(self.graph.edges[self.dfsStackKeys[curDepth]]) - 1:
                self.dfsStack[curDepth] = 0
                var node: String = self.dfsStackKeys.pop()
                curDepth = len(self.dfsStackKeys) - 1
                
            if curDepth < 0:
                self.sampleArchitecturesEnd = True 
                return False
            else:
                self.dfsStack[curDepth] = self.dfsStack[curDepth] + 1
                var curDfsNode: String = self.graph.edges[self.dfsStackKeys[curDepth]][self.dfsStack[curDepth]]
                while curDfsNode != 'output':
                    self.dfsStackKeys.append(curDfsNode)
                    curDepth = len(self.dfsStackKeys) - 1
                    self.dfsStack[curDepth] = 0
                    curDfsNode = self.graph.edges[curDfsNode][0]
        else:
            self.dfsStack[curDepth] = self.dfsStack[curDepth] + 1
        self.sampleArchitecturesEnd = False
        return True
        
    def nextSampleArchitecture(inout self) -> Graph:
        if self.sampleArchitecturesEnd == True:
            self.initDfsStack()
        
        var dfsSample: List[Int]      
        if self.incSampleArchitecture():  
            dfsSample = self.dfsStack
            sampleGraph = self.sampleArchitecture(dfsSample)
            return sampleGraph
        else:
            dfsSample = self.dfsStack
            sampleGraph = self.sampleArchitecture(dfsSample)
            return sampleGraph
        
    def sampleArchitecture(inout self, sample: List[Int]) -> Graph:
        var nodeName: String = 'input'
        self.sample = Graph()        
        self.sample.nodes[nodeName] = self.graph.nodes[nodeName]   
        self.sample.edges[nodeName] = self.graph.edges[nodeName][sample[0]]
        nodeName = self.graph.edges[nodeName][sample[0]]
        var ind: Int = 1
 
        while nodeName != 'output':
            if ind >= len(sample):
                print("uh-oh")
            self.sample.nodes[nodeName] = self.graph.nodes[nodeName]
            self.sample.edges[nodeName] = self.graph.edges[nodeName][sample[ind]]
            nodeName = self.graph.edges[nodeName][sample[ind]]
            ind += 1
        
        self.sample.nodes[nodeName] = self.graph.nodes[nodeName]
        self.sample.edges[nodeName] = List[String]()

        return self.sample
        