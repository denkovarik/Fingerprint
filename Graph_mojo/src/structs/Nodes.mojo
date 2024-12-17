from collections import Optional
from python import Python


struct NodeType:    
    # The value for the enum
    var INPUT: String
    var OUTPUT: String
    var CONVOLUTION: String
    var NORMALIZATION: String
    var POOLING: String
    var FLATTEN: String
    var LINEAR: String
    var ACTIVATION: String
    var NONE: String
    var value: String
    
    fn __init__(inout self, theType: String):
        self.INPUT = 'input'  
        self.OUTPUT = 'output'
        self.CONVOLUTION = 'convolution'
        self.NORMALIZATION = 'normalization'
        self.POOLING = 'pooling'
        self.FLATTEN = 'flatten'
        self.LINEAR = 'linear'
        self.ACTIVATION = 'activation'
        self.NONE = 'none'
        
        self.value = self.NONE
        self.value = self.getValidNodeType(theType)
            
    fn getValidNodeType(self, theType: String) -> String:
        if theType == self.INPUT:
            return self.INPUT
        elif theType == self.OUTPUT:
            return self.OUTPUT
        elif theType == self.CONVOLUTION:
            return self.CONVOLUTION
        elif theType == self.NORMALIZATION:
            return self.NORMALIZATION
        elif theType == self.POOLING:
            return self.POOLING
        elif theType == self.FLATTEN:
            return self.FLATTEN
        elif theType == self.LINEAR:
            return self.LINEAR
        elif theType == self.ACTIVATION:
            return self.ACTIVATION
        else:
            return self.NONE
        
      
struct NormalizationType:    
    # The value for the enum
    var NO_NORM: String
    var BATCH_NORM: String
    var NONE: String
    var value: String
    
    fn __init__(inout self, theType: String):
        self.NO_NORM = 'noNorm'  
        self.BATCH_NORM = 'batchNorm'
        self.NONE = 'none'
        
        self.value = self.NONE
        self.value = self.getValidNodeType(theType)
            
    fn getValidNodeType(self, theType: String) -> String:
        if theType == self.NO_NORM:
            return self.NO_NORM
        elif theType == self.BATCH_NORM:
            return self.BATCH_NORM
        else:
            return self.NONE
            
        
struct PoolingType:    
    # The value for the enum
    var NO_POOLING: String
    var MAX_POOLING: String
    var NONE: String
    var value: String
    
    fn __init__(inout self, theType: String):
        self.NO_POOLING = 'noPooling'  
        self.MAX_POOLING = 'maxPooling'
        self.NONE = 'none'
        
        self.value = self.NONE
        self.value = self.getValidNodeType(theType)
            
    fn getValidNodeType(self, theType: String) -> String:
        if theType == self.NO_POOLING:
            return self.NO_POOLING
        elif theType == self.MAX_POOLING:
            return self.MAX_POOLING
        else:
            return self.NONE
            
            
struct ActivationType:    
    var RELU: String    # Value for Relu activation
    var NONE: String    # Value for No activation type
    var value: String   # The value for the enum
    
    fn __init__(inout self, theType: String):
        self.RELU = 'reluActivation'  
        self.NONE = 'none'
        
        self.value = self.NONE
        self.value = self.getValidNodeType(theType)
            
    fn getValidNodeType(self, theType: String) -> String:
        if theType == self.RELU:
            return self.RELU
        else:
            return self.NONE
            
            
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
        