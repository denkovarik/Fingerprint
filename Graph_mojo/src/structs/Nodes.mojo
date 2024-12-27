from collections import Optional
from python import Python, PythonObject


# Enums

@value
@register_passable("trivial")
struct NodeType:
    var value: Int
    
    alias invalid = NodeType(0)
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
  

@value
@register_passable("trivial")
struct PoolingType:
    var value: Int
    
    alias invalid = PoolingType(0)
    alias NO_POOLING = PoolingType(1) 
    alias MAX_POOLING = PoolingType(2)
    

@value
@register_passable("trivial")
struct ActivationType:
    var value: Int
    
    alias invalid = ActivationType(0)
    alias RELU = ActivationType(1) 
    alias NONE = ActivationType(2)
            
            
struct InputNode:
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
        
               
struct OutputNode:
    var displayName: String
    var name: String
    
    fn __init__(inout self):
        self.name = 'output'
        self.displayName = 'Output'
        
        
struct NormalizationNode:
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
            
      
struct PoolingNode:
    var displayName: String
    var name: String
    var poolingType: PoolingType
    var kernelSize: Int
    var stride: Int
    
    fn __init__(inout self, name: String, poolingType: PoolingType):
        self.name = name 
        self.displayName = 'No Pooling'
        self.poolingType = poolingType
        self.kernelSize = 2
        self.stride = 2
        if poolingType.value == PoolingType.MAX_POOLING.value:
            self.displayName = 'Max Pooling'
            
            
struct ActivationNode:
    var displayName: String
    var name: String
    var activationType: ActivationType

    fn __init__(inout self, name: String, activationType: ActivationType):
        self.activationType = activationType
        self.name = name
        self.displayName = "None"
        if self.activationType.value == activationType.NONE.value:
            self.displayName = 'No Activation'
        elif self.activationType.value == activationType.RELU.value:
            self.displayName = 'Relu Activation'


struct FlattenNode:
    var displayName: String
    var name: String
    
    def __init__(inout self, name: String):
        self.name = name
        self.displayName = 'Flatten'