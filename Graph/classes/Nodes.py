from enum import Enum

class NodeType(Enum):
    INPUT = 'input'
    OUTPUT = 'output'
    CONVOLUTION = 'convolution'
    NORMALIZATION = 'normalization'
    POOLING = 'pooling'
    FLATTEN = 'flatten'
    LINEAR = 'linear'
    ACTIVATION = 'activation'


class NormalizationType(Enum):
    NO_NORM = 'noNorm'
    BATCH_NORM = 'batchNorm'


class PoolingType(Enum):
    NO_POOLING = 'noPooling'
    MAX_POOLING = 'maxPooling'


class ActivationType(Enum):
    RELU = 'reluActivation'
    LINEAR = 'linearActivation'


class Node:
    def __init__(self, name, displayName):
        self.name = name
        self.displayName = displayName
                                    
    def __str__(self):
        return self.displayName
                                                    
    def __repr__(self):
        return self.displayName


class InputNode(Node):
    def __init__(self, numChannels):
        self.name = 'input'
        self.displayName = 'Input(numChannels=' + str(numChannels) + ')'
        self.numChannels = numChannels


class OutputNode(Node):
    def __init__(self):
        self.name = 'output'
        self.displayName = 'Output'


class NormalizationNode(Node):
    def __init__(self, name, normalizationType):
        self.name = name
        self.displayName = 'No Normalization'
        self.normalizationType = normalizationType
        if normalizationType == NormalizationType.BATCH_NORM:
            self.displayName = 'Batch Normalization'


class PoolingNode(Node):
    def __init__(self, name, poolingType):
        self.name = name 
        self.displayName = 'No Pooling'
        self.poolingType = poolingType
        if poolingType == PoolingType.MAX_POOLING:
            self.displayName = 'Max Pooling'


class ConvolutionalNode(Node):
    def __init__(self, name, kernelSize, maxNumInputChannels, numOutputChannels, layer):
        typeErrMsg = "Kernel Size must be either an integer, a tuple of "
        typeErrMsg += "integers, or a list 2 integers"

        if isinstance(kernelSize, int):
            # Kernel size as just an int indicates a kernel size of 
            # kernelSize x kernelSize
            self.kernelSize = (kernelSize, kernelSize)
        elif isinstance(kernelSize, list) or isinstance(kernelSize, tuple):
            if (len(kernelSize) == 2 and isinstance(kernelSize[0], int)  
            and isinstance(kernelSize[1], int)):
                self.kernelSize = (kernelSize[0], kernelSize[1])
            else:
                raise TypeError(typeErrMsg)
        else:
            raise TypeError(typeErrMsg) 
        
        self.name = name
        self.layer = layer
        self.displayName = str(self.kernelSize[0]) + 'x' + str(self.kernelSize[1]) 
        self.displayName += ' Conv(oc=' + str(numOutputChannels) + ')'
        self.maxNumInputChannels = maxNumInputChannels
        self.numOutputChannels = numOutputChannels


class FlattenNode(Node):
    def __init__(self, name):
        self.name = name
        self.displayName = 'Flatten'


class LinearNode(Node):
    def __init__(self, name, maxNumInFeatures, numOutFeatures, layer):
        self.name = name
        self.layer = layer
        self.maxNumInFeatures = maxNumInFeatures
        self.numOutFeatures = numOutFeatures
        self.displayName = 'Linear(of=' + str(numOutFeatures) + ')'


class ActivationNode(Node):
    def __init__(self, name, activationType):
        self.activationType = activationType
        self.name = name
        if self.activationType == activationType.LINEAR:
            self.displayName = 'Linear Activation'
        elif self.activationType == activationType.RELU:
            self.displayName = 'Relu Activation'
        else:
            raise ValueError(f"Unknown activation type: {activationType}")


class NodeFactory:
    def createNode(self, nodeType, *args, **kwargs):
        if nodeType == NodeType.INPUT:
            return InputNode(*args, **kwargs)
        if nodeType == NodeType.OUTPUT:
            return OutputNode(*args, **kwargs)
        if nodeType == NodeType.NORMALIZATION:
            return NormalizationNode(*args, **kwargs)
        if nodeType == NodeType.POOLING:
            return PoolingNode(*args, **kwargs)
        if nodeType == NodeType.CONVOLUTION:
            return ConvolutionalNode(*args, **kwargs)
        if nodeType == NodeType.FLATTEN:
            return FlattenNode(*args, **kwargs)
        if nodeType == NodeType.LINEAR:
            return LinearNode(*args, **kwargs)
        if nodeType == NodeType.ACTIVATION:
            return ActivationNode(*args, **kwargs)

        raise ValueError(f"Unknown node type: {nodeType}")
