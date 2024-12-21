from collections import Optional
from python import Python

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
    

    fn __init__(inout self) raises:
        torch = Python.import_module("torch")
        
        self.displayName = 'Input'
        #if not isinstance(inputShape, torch.Size) or len(inputShape) != 4:
        #    raise ValueError("inputShape must be a torch.Size of length 4")
        #self.name = 'input'
        #self.numChannels = inputShape[1]
        #self.displayName = 'Input(numChannels=' + str(self.numChannels) + ')'
        #self.displayName = 'Input(numChannels=' + str(self.numChannels) + ')'
        #self.inputShape = inputShape
        