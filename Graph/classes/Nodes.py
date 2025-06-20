from enum import Enum
import torch
import torch.nn as nn
from torch.profiler import profile, ProfilerActivity, record_function
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
    NONE = 'noActivation'


class Node:
    def __init__(self, name="node", displayName="Node"):
        self.name = name
        self.displayName = displayName
        self.pytorchLayerId = None
        self.pytorchLayer = None

    def __call__(self, x):
        return self.forward(x)

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

    def setPytochLayer(node, sharedLayer):
        if not isinstance(node, Node):
            raise Exception("Param node must be of type Node")

        if isinstance(node, ConvolutionalNode):
            pass
        elif isinstance(node, LinearNode):
            pass
        return None


class InputNode(Node):
    def __init__(self, inputShape):
        super().__init__()
        if not isinstance(inputShape, torch.Size) or len(inputShape) != 4:
            raise ValueError("inputShape must be a torch.Size of length 4")
        self.name = 'input'
        self.numChannels = inputShape[1]
        self.inputShape = inputShape
        self.displayName = f'Input(numChannels={self.numChannels})'
        
    def __eq__(self, other):
        if isinstance(other, InputNode):
            return self.input_shape == other.input_shape
        return False

    def __hash__(self):
        # Create a hashable tuple from the instance's attributes
        return hash((self.name, tuple(self.inputShape)))
        
    def getLayer(self, inputShape):
        return PassThrough(displayName=self.displayName)


class OutputNode(Node):
    def __init__(self):
        super().__init__() 
        self.name = 'output'
        self.displayName = 'Output'
        
    def __eq__(self, other):
        if isinstance(other, OutputNode):
            return True 
        return False

    def __hash__(self):
        return hash('OutputNode')
        
    def getLayer(self, inputShape):
        return PassThrough(displayName=self.displayName)
    

class PassThrough(nn.Module):
    def __init__(self, displayName='PassThrough'):
        super(PassThrough, self).__init__()   
        self.displayName = displayName

    def __str__(self):
        return self.displayName
        
    def __repr__(self):
        return self.displayName
    
    def forward(self, x):
        return x   
    
    def to(self, device):
        return self


class NormalizationNode(Node):
    def __init__(self, name, normalizationType, numFeatures, pytorchLayerId):
        super().__init__(name, 'No Normalization')
        self.normalizationType = normalizationType
        self.numFeatures = numFeatures
        if normalizationType == NormalizationType.BATCH_NORM:
            self.displayName = 'Batch Normalization'
            self.pytorchLayer = nn.BatchNorm2d(self.numFeatures)

    def __eq__(self, other):
        if isinstance(other, NormalizationNode):
            return (self.name == other.name and
                    self.normalizationType == other.normalizationType and
                    self.numFeatures == other.numFeatures)
        return False

    def __hash__(self):
        return hash((self.name, self.normalizationType, self.numFeatures))
        
    def getLayer(self, inputShape):
        with record_function("getNormalizationNode"):
            bn = nn.BatchNorm2d(inputShape[1])
            if self.normalizationType == NormalizationType.NO_NORM:
                return PassThrough(displayName=self.displayName)
            return bn        
            
    def parameters(self, recurse: bool = True):
        if self.pytorchLayer:
            return self.pytorchLayer.parameters(recurse=recurse)
        else:
            return iter([]) 


class PoolingNode(Node):
    def __init__(self, name, poolingType):
        super().__init__() 
        self.name = name 
        self.displayName = 'No Pooling'
        self.poolingType = poolingType
        self.kernelSize = 2
        self.stride = 2
        if poolingType == PoolingType.MAX_POOLING:
            self.displayName = 'Max Pooling'
                
    def __eq__(self, other):
        if isinstance(other, PoolingNode):
            return (self.name == other.name and
                    self.poolingType == other.poolingType and
                    self.kernelSize == other.kernelSize and
                    self.stride == other.stride)
        return False

    def __hash__(self):
        return hash((self.name, self.poolingType, self.kernelSize, self.stride))

    def getLayer(self, inputShape):
        with record_function("getPoolingNode"):
            if self.poolingType == PoolingType.MAX_POOLING:
                return nn.MaxPool2d(self.kernelSize, self.stride)
            return PassThrough(self.displayName)


class ConvolutionalNode(Node):
    def __init__(self, name, kernel_size, maxNumInputChannels, 
                 maxNumOutputChannels, numOutputChannels, layer, pytorchLayerId):
        super().__init__() 
        typeErrMsg = "Kernel Size must be either an integer, a tuple of "
        typeErrMsg += "integers, or a list 2 integers"

        if isinstance(kernel_size, int):
            # Kernel size as int => kernel_size x kernel_size
            self.kernel_size = (kernel_size, kernel_size)
        elif isinstance(kernel_size, list) or isinstance(kernel_size, tuple):
            if (len(kernel_size) == 2 and isinstance(kernel_size[0], int)  
            and isinstance(kernel_size[1], int)):
                self.kernel_size = (kernel_size[0], kernel_size[1])
            else:
                raise TypeError(typeErrMsg)
        else:
            raise TypeError(typeErrMsg) 
        
        self.name = name
        self.displayName = str(self.kernel_size[0]) + 'x' + str(self.kernel_size[1]) 
        self.displayName += ' Conv(oc=' + str(numOutputChannels) + ')'
        self.layer = layer
        self.pytorchLayerId = pytorchLayerId
        self.maxNumInputChannels = maxNumInputChannels
        self.maxNumOutputChannels = maxNumOutputChannels
        self.numOutputChannels = numOutputChannels
        self.sharedConv2DLayer = None
        
    def __eq__(self, other):
        if isinstance(other, ConvolutionalNode):
            return (self.name == other.name and
                    self.kernel_size == other.kernel_size and
                    self.maxNumInputChannels == other.maxNumInputChannels and
                    self.maxNumOutputChannels == other.maxNumOutputChannels and
                    self.numOutputChannels == other.numOutputChannels)
        return False

    def __hash__(self):
        return hash((self.name, self.kernel_size, self.maxNumInputChannels, 
                     self.maxNumOutputChannels, self.numOutputChannels))

    def constructLayer(self):
        return SharedConv2d(kernel_size=self.kernel_size, 
                            in_channels=self.maxNumInputChannels, 
                            out_channels=self.maxNumOutputChannels)

    def getLayer(self, inputShape):
        with record_function("getConvolutionalNode"):
            return self

    def forward(self, x):
        return self.pytorchLayer(x, x.shape[1], self.numOutputChannels)
        
    def parameters(self, recurse: bool = True):
        # Yield the parameters of the pytorchLayer
        for param in self.pytorchLayer.parameters(recurse=recurse):
            yield param

    def setSharedLayer(self, pytorchLayer):
        self.pytorchLayer = pytorchLayer
        
    def to(self, device):
        self.pytorchLayer = self.pytorchLayer.to(device)  # Ensure the convolution layer is also moved
        return self 


class FlattenNode(Node):
    def __init__(self, name):
        super().__init__() 
        self.name = name
        self.displayName = 'Flatten'
        
    def __eq__(self, other):
        if isinstance(other, FlattenNode):
            return self.name == other.name  # All FlattenNode instances with the same name are considered equal
        return False

    def __hash__(self):
        return hash(('FlattenNode', self.name))

    def getLayer(self, inputShape):
        with record_function("getFlattenNode"):
            return nn.Flatten()


class LinearNode(Node):
    def __init__(self, name, maxNumInFeatures, maxNumOutFeatures, 
                 numOutFeatures, layer, pytorchLayerId):
        super().__init__() 
        self.name = name
        self.displayName = 'Linear(of=' + str(numOutFeatures) + ')'
        self.layer = layer
        self.pytorchLayerId = pytorchLayerId
        self.maxNumInFeatures = maxNumInFeatures
        self.maxNumOutFeatures = maxNumOutFeatures
        self.numOutFeatures = numOutFeatures
        
    def __eq__(self, other):
        if isinstance(other, LinearNode):
            return (self.name == other.name and
                    self.maxNumInFeatures == other.maxNumInFeatures and
                    self.maxNumOutFeatures == other.maxNumOutFeatures and
                    self.numOutFeatures == other.numOutFeatures)
        return False

    def __hash__(self):
        return hash((self.name, self.maxNumInFeatures, self.maxNumOutFeatures, self.numOutFeatures))

    def constructLayer(self):
        return SharedLinear(self.maxNumInFeatures, self.maxNumOutFeatures)

    def forward(self, x):
        return self.pytorchLayer(x, x.shape[1], self.numOutFeatures)

    def getLayer(self, inputShape):
        with record_function("getLinearNode"):
            return self
            
    def parameters(self, recurse: bool = True):
        # Yield the parameters of the pytorchLayer
        for param in self.pytorchLayer.parameters(recurse=recurse):
            yield param

    def setSharedLayer(self, pytorchLayer):
        self.pytorchLayer = pytorchLayer
           
    def to(self, device):
        self.pytorchLayer = self.pytorchLayer.to(device)  # Ensure the convolution layer is also moved
        return self 


class ActivationNode(Node):
    def __init__(self, name, activationType):
        super().__init__() 
        self.activationType = activationType
        self.name = name
        if self.activationType == activationType.NONE:
            self.displayName = 'No Activation'
        elif self.activationType == activationType.RELU:
            self.displayName = 'Relu Activation'
        else:
            raise ValueError(f"Unknown activation type: {activationType}")

    def __eq__(self, other):
        if isinstance(other, ActivationNode):
            return (self.name == other.name and
                    self.activationType == other.activationType)
        return False

    def __hash__(self):
        return hash((self.name, self.activationType))
        
    def getLayer(self, inputShape):
        with record_function("getActivationNode"):
            return nn.ReLU()


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
