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