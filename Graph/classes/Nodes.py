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
    NO_POOLING = 'none'
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
    def __init__(self, name):
        self.displayName = 'No Normalization'
        if 'batchNorm' in name:
            self.displayName = 'Batch Normalization'
        self.name = name


class PoolingNode(Node):
    def __init__(self, name):
        self.displayName = 'No Pooling'
        if 'maxPooling' in name:
            self.displayName = 'Max Pooling'
        self.name = name 


class ConvolutionalNode(Node):
    def __init__(self, name, kernelSize, outputChannels):
        self.name = name
        self.displayName = str(kernelSize[0]) + 'x' + str(kernelSize[1]) + ' Conv(oc=' + str(outputChannels) + ')'
        self.kernelSize = kernelSize
        self.outputChannels = outputChannels


class FlattenNode(Node):
    def __init__(self, name):
        self.name = name
        self.displayName = 'Flatten'


class LinearNode(Node):
    def __init__(self, name, outFeatures):
        self.name = name
        self.outFeatures = outFeatures
        self.displayName = 'Linear(of=' + str(outFeatures) + ')'


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
