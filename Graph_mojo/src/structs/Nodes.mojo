from collections import Optional
from python import Python, PythonObject
from structs.SharedConv2d import SharedConv2d
from utils import Variant


# Enums

@value
@register_passable("trivial")
struct NodeType:
    var value: Int
    
    alias NONE = NodeType(0)
    alias INPUT = NodeType(1) 
    alias OUTPUT = NodeType(2)
    alias CONVOLUTION = NodeType(3)
    alias NORMALIZATION = NodeType(4)
    alias POOLING = NodeType(5)
    alias FLATTEN = NodeType(6)
    alias LINEAR = NodeType(7)
    alias ACTIVATION = NodeType(8)
    
            
@value
@register_passable("trivial")
struct NormalizationType:
    var value: Int
    
    alias invalid = NormalizationType(0)
    alias NO_NORM = NormalizationType(1) 
    alias BATCH_NORM = NormalizationType(2)
    
    fn __init__(inout self):
        pass
    
    fn __eq__(self, other: NormalizationType) -> Bool:
        if self.value == other.value:
            return True
        return False
        
    fn __ne__(self, other: NormalizationType) -> Bool:
        if self.value != other.value:
            return True
        return False
  

@value
@register_passable("trivial")
struct PoolingType:
    var value: Int
    
    alias invalid = PoolingType(0)
    alias NO_POOLING = PoolingType(1) 
    alias MAX_POOLING = PoolingType(2)
    
    fn __init__(inout self):
        pass
    
    fn __eq__(self, other: PoolingType) -> Bool:
        if self.value == other.value:
            return True
        return False
        
    fn __ne__(self, other: PoolingType) -> Bool:
        if self.value != other.value:
            return True
        return False
    

@value
@register_passable("trivial")
struct ActivationType:
    var value: Int
    
    alias invalid = ActivationType(0)
    alias RELU = ActivationType(1) 
    alias NONE = ActivationType(2)
            
            
# Define the trait for Node
trait NodeTrait:
    def forward(inout self, x: PythonObject) -> PythonObject:
        pass
    

# Yankee Navy Assholes!
alias NodeVariant = Variant[InputNode, OutputNode, NormalizationNode, PoolingNode, ActivationNode, FlattenNode, LinearNode, ConvolutionalNode]

@value
struct Node(NodeTrait):
    # ● 
    # ● 
    var node: NodeVariant
    var nodeType: NodeType

    fn __init__(inout self, theNode: NodeVariant) raises:
        self.nodeType = NodeType.NONE
        self.node = theNode
        self.nodeType = self.getNodeType()
        
    def getNodeType(inout self) -> NodeType:
        if self.node.isa[InputNode]():
            return NodeType.INPUT
        if self.node.isa[OutputNode]():
            return NodeType.OUTPUT
        if self.node.isa[ConvolutionalNode]():
            return NodeType.CONVOLUTION
        if self.node.isa[NormalizationNode]():
            return NodeType.NORMALIZATION
        if self.node.isa[PoolingNode]():
            return NodeType.POOLING
        if self.node.isa[FlattenNode]():
            return NodeType.FLATTEN
        if self.node.isa[LinearNode]():
            return NodeType.LINEAR
        if self.node.isa[ActivationNode]():
            return NodeType.ACTIVATION
        return NodeType.NONE
        
    def forward(inout self, x: PythonObject) -> PythonObject:
        var out = x
        self.nodeType = self.getNodeType()
        
        if self.nodeType.value == NodeType.INPUT.value:
            out = self.node[InputNode].forward(x)
        elif self.nodeType.value == NodeType.OUTPUT.value:
            out = self.node[OutputNode].forward(x)
        elif self.nodeType.value == NodeType.CONVOLUTION.value:
            out = self.node[ConvolutionalNode].forward(x)
        elif self.nodeType.value == NodeType.NORMALIZATION.value:
            out = self.node[NormalizationNode].forward(x)
        elif self.nodeType.value == NodeType.POOLING.value:
            out = self.node[PoolingNode].forward(x)
        elif self.nodeType.value == NodeType.FLATTEN.value:
            out = self.node[FlattenNode].forward(x)
        elif self.nodeType.value == NodeType.LINEAR.value:
            out = self.node[LinearNode].forward(x)
        elif self.nodeType.value == NodeType.ACTIVATION.value:
            out = self.node[ActivationNode].forward(x)
        
        return out
    

@value
struct InputNode(NodeTrait):
    var displayName: String
    var inputShape: PythonObject
    var numChannels: Int
    var name: String

    fn __init__(inout self, inputShape: PythonObject) raises:
        torch = Python.import_module("torch")      
        self.inputShape = inputShape
        self.name = 'input'
        self.numChannels = inputShape[1]
        self.displayName = 'Input(numChannels=' + str(self.numChannels) + ')'
               
    def forward(inout self, x: PythonObject) -> PythonObject:
        return x
     

@value     
struct OutputNode(NodeTrait):
    var displayName: String
    var name: String
    
    fn __init__(inout self):
        self.name = 'output'
        self.displayName = 'Output'
        
    def forward(inout self, x: PythonObject) -> PythonObject:
        return x
        

@value  
struct NormalizationNode(NodeTrait):
    var displayName: String
    var name: String
    var normalizationType: NormalizationType
    var numFeatures: Int
    var pytorchLayerId: PythonObject
    var pytorchLayer: PythonObject

    fn __init__(inout self, name: String, normalizationType: NormalizationType, numFeatures: Int, pytorchLayerId: PythonObject) raises:
        nn = Python.import_module("torch.nn")
        self.pytorchLayer = None
        self.name = name
        self.displayName = 'No Normalization'
        self.normalizationType = normalizationType
        self.numFeatures = numFeatures
        self.pytorchLayerId = pytorchLayerId
        if normalizationType == NormalizationType.BATCH_NORM:
            self.displayName = 'Batch Normalization'
            self.pytorchLayer = nn.BatchNorm2d(self.numFeatures)
            
    fn __str__(inout self) -> String:
        var strRep: String = 'NoNorm2d()'
        if self.normalizationType != NormalizationType.NO_NORM:
            strRep = str(self.pytorchLayer)
        return strRep
            
    def forward(inout self, x: PythonObject) -> PythonObject:
        if self.normalizationType == NormalizationType.NO_NORM:
            return x
        return self.pytorchLayer.forward(x)
            

@value      
struct PoolingNode(NodeTrait):
    var displayName: String
    var name: String
    var poolingType: PoolingType
    var kernelSize: Int
    var stride: Int
    var pytorchLayer: PythonObject
    
    fn __init__(inout self, name: String, poolingType: PoolingType) raises:
        self.name = name 
        self.displayName = 'No Pooling'
        self.poolingType = poolingType
        self.kernelSize = 2
        self.stride = 2
        nn = Python.import_module("torch.nn")
        self.pytorchLayer = None
        if poolingType.value == PoolingType.MAX_POOLING.value:
            self.displayName = 'Max Pooling'
            self.pytorchLayer = nn.MaxPool2d(self.kernelSize, self.stride)
            
    fn __str__(inout self) -> String:
        var strRep: String = 'NoPooling2d()'
        if self.poolingType != PoolingType.NO_POOLING:
            strRep = str(self.pytorchLayer)
        return strRep
            
    def forward(inout self, x: PythonObject) -> PythonObject:
        if self.poolingType.value == PoolingType.NO_POOLING.value:
            return x
        return self.pytorchLayer.forward(x)
        

@value            
struct ActivationNode(NodeTrait):
    var displayName: String
    var name: String
    var activationType: ActivationType
    var pytorchLayer: PythonObject

    def __init__(inout self, name: String, activationType: ActivationType):
        self.activationType = activationType
        self.name = name
        self.displayName = "None"
        nn = Python.import_module("torch.nn")
        self.pytorchLayer = None
        self.displayName = 'No Activation'
        if self.activationType.value == activationType.RELU.value:
            self.displayName = 'Relu Activation'
            self.pytorchLayer = nn.ReLU()
            
    def forward(inout self, x: PythonObject) -> PythonObject:
        if self.activationType.value == ActivationType.NONE.value:
            return x
        return self.pytorchLayer.forward(x)
        

@value
struct FlattenNode(NodeTrait):
    var displayName: String
    var name: String
    var pytorchLayer: PythonObject
    
    def __init__(inout self, name: String):
        self.name = name
        self.displayName = 'Flatten'
        nn = Python.import_module("torch.nn")
        self.pytorchLayer = nn.Flatten()
            
    def forward(inout self, x: PythonObject) -> PythonObject:
        return self.pytorchLayer(x)   
        
  
@value   
struct ConvolutionalNode(NodeTrait):
    var kernel_size: Int
    var name: String
    var displayName: String
    var layer: Int
    var pytorchLayerId: Int
    var maxNumInputChannels: Int
    var maxNumOutputChannels: Int
    var numOutputChannels: Int
    var pytorchLayer: PythonObject

    fn __init__(inout self, name: String, kernel_size: Int, maxNumInputChannels: Int, 
                 maxNumOutputChannels: Int, numOutputChannels: Int, layer: Int, pytorchLayerId: Int):        
        self.kernel_size = kernel_size
        self.name = name
        self.displayName = str(self.kernel_size) + 'x' + str(self.kernel_size) 
        self.displayName += ' Conv(oc=' + str(numOutputChannels) + ')'
        self.layer = layer
        self.pytorchLayerId = pytorchLayerId
        self.maxNumInputChannels = maxNumInputChannels
        self.maxNumOutputChannels = maxNumOutputChannels
        self.numOutputChannels = numOutputChannels
        self.pytorchLayer = None

    def constructLayer(inout self):
        return SharedConv2d(kernel_size=self.kernel_size, 
                            in_channels=self.maxNumInputChannels, 
                            out_channels=self.maxNumOutputChannels)

    def forward(inout self, x: PythonObject) -> PythonObject:
        return self.pytorchLayer(x, x.shape[1], self.numOutputChannels)

    def setSharedLayer(inout self, pytorchLayer: PythonObject):
        self.pytorchLayer = pytorchLayer
           
    def to(inout self, device: PythonObject):
        self.pytorchLayer = self.pytorchLayer.to(device)  # Ensure the convolution layer is also moved
        return self 
        

@value     
struct LinearNode(NodeTrait):
    var name: String
    var displayName: String
    var layer: Int
    var pytorchLayerId: Int
    var maxNumInFeatures: Int
    var maxNumOutFeatures: Int
    var numOutFeatures: Int
    var pytorchLayer: PythonObject
        
    fn __init__(inout self, name: String, maxNumInFeatures: Int, maxNumOutFeatures: Int, 
                 numOutFeatures: Int, layer: Int, pytorchLayerId: Int):
        self.name = name
        self.displayName = 'Linear(of=' + str(numOutFeatures) + ')'
        self.layer = layer
        self.pytorchLayerId = pytorchLayerId
        self.maxNumInFeatures = maxNumInFeatures
        self.maxNumOutFeatures = maxNumOutFeatures
        self.numOutFeatures = numOutFeatures
        self.pytorchLayer = None

    def constructLayer(inout self):
        return SharedLinear(self.maxNumInFeatures, self.maxNumOutFeatures)

    def forward(inout self, x: PythonObject) -> PythonObject:
        return self.pytorchLayer(x, x.shape[1], self.numOutFeatures)

    def setSharedLayer(inout self, pytorchLayer: PythonObject):
        self.pytorchLayer = pytorchLayer
             
    def to(inout self, device: PythonObject):
        self.pytorchLayer = self.pytorchLayer.to(device)  # Ensure the convolution layer is also moved
        return self 
        
        
