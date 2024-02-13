from enum import Enum
import torch
import torch.nn as nn
from classes.SharedConv2d import SharedConv2d
from classes.SharedLinear import SharedLinear


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
    def __init__(self, name="node", displayName="Node"):
        self.name = name
        self.displayName = displayName
        self.pytorchLayerId = None


    def __str__(self):
        return self.displayName

    def __repr__(self):
        return self.displayName


    def getPytochLayer(node):
        if not isinstance(node, Node):
            raise Exception("Param node must be of type Node")

        if isinstance(node, ConvolutionalNode):
            pass
        elif isinstance(node, LinearNode):
            pass
        return None


class InputNode(Node):
    def __init__(self, numChannels):
        super().__init__() 
        self.name = 'input'
        self.displayName = 'Input(numChannels=' + str(numChannels) + ')'
        self.displayName = 'Input(numChannels=' + str(numChannels) + ')'
        self.numChannels = numChannels


class OutputNode(Node):
    def __init__(self):
        super().__init__() 
        self.name = 'output'
        self.displayName = 'Output'


class NormalizationNode(Node):
    def __init__(self, name, normalizationType):
        super().__init__() 
        self.name = name
        self.displayName = 'No Normalization'
        self.normalizationType = normalizationType
        if normalizationType == NormalizationType.BATCH_NORM:
            self.displayName = 'Batch Normalization'


class PoolingNode(Node):
    def __init__(self, name, poolingType):
        super().__init__() 
        self.name = name 
        self.displayName = 'No Pooling'
        self.poolingType = poolingType
        if poolingType == PoolingType.MAX_POOLING:
            self.displayName = 'Max Pooling'


class ConvolutionalNode(Node):
    def __init__(self, name, kernelSize, maxNumInputChannels, 
                 maxNumOutputChannels, numOutputChannels, layer, pytorchLayerId):
        super().__init__() 
        typeErrMsg = "Kernel Size must be either an integer, a tuple of "
        typeErrMsg += "integers, or a list 2 integers"

        if isinstance(kernelSize, int):
            # Kernel size as int => kernelSize x kernelSize
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
        self.displayName = str(self.kernelSize[0]) + 'x' + str(self.kernelSize[1]) 
        self.displayName += ' Conv(oc=' + str(numOutputChannels) + ')'
        self.layer = layer
        self.pytorchLayerId = pytorchLayerId
        self.maxNumInputChannels = maxNumInputChannels
        self.maxNumOutputChannels = numOutputChannels
        self.numOutputChannels = numOutputChannels


    def getPytorchLayer(self):
        return SharedConv2d(kernel_size=self.kernelSize, 
                            in_channels=self.maxNumInputChannels, 
                            out_channels=self.maxNumOutputChannels)


class FlattenNode(Node):
    def __init__(self, name):
        super().__init__() 
        self.name = name
        self.displayName = 'Flatten'


class LinearNode(Node):
    def __init__(self, name, maxNumInFeatures, maxNumOutFeatures, 
                 numOutFeatures, layer, pytorchLayerId):
        super().__init__() 
        self.name = name
        self.displayName = 'Linear(of=' + str(numOutFeatures) + ')'
        self.layer = layer
        self.pytorchLayerId = pytorchLayerId
        self.maxNumInFeatures = maxNumInFeatures
        self.maxNumOutFeatures = numOutFeatures
        self.numOutFeatures = numOutFeatures


    def getPytorchLayer(self):
        return SharedLinear(maxInFeatures=self.maxNumInFeatures, 
                            maxOutFeatures=self.maxNumOutFeatures)


class ActivationNode(Node):
    def __init__(self, name, activationType):
        super().__init__() 
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
