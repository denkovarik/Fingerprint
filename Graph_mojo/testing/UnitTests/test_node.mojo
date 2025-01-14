# Content of test_quickstart.mojo
from testing import assert_equal, assert_not_equal, assert_true, assert_false
from python import Python, PythonObject
from structs.Nodes import NodeType, NormalizationType, PoolingType, ActivationType
from structs.Nodes import Node, InputNode, OutputNode, NormalizationNode, PoolingNode, ActivationNode, FlattenNode
from structs.Nodes import ConvolutionalNode, LinearNode, NodeTrait


def test_execution():
    # Just tests running a mojo test
    assert_equal(0, 0)

def test_nodeTypes():
    """
    Tests the ability to use NodeType.
    """	    
    # INPUT
    var nodeType = NodeType.INPUT
    assert_equal(nodeType.value, NodeType.INPUT.value)
        
    # OUTPUT
    nodeType = NodeType.OUTPUT
    assert_equal(nodeType.value, NodeType.OUTPUT.value)
        
    # CONVOLUTION
    nodeType = NodeType.CONVOLUTION
    assert_equal(nodeType.value, NodeType.CONVOLUTION.value)
        
    # NORMALIZATION
    nodeType = NodeType.NORMALIZATION
    assert_equal(nodeType.value, NodeType.NORMALIZATION.value)
        
    # POOLING
    nodeType = NodeType.POOLING
    assert_equal(nodeType.value, NodeType.POOLING.value)
        
    # FLATTEN
    nodeType = NodeType.FLATTEN
    assert_equal(nodeType.value, NodeType.FLATTEN.value)
        
    # LINEAR
    nodeType = NodeType.LINEAR
    assert_equal(nodeType.value, NodeType.LINEAR.value)
            
    # ACTIVATION
    nodeType = NodeType.ACTIVATION
    assert_equal(nodeType.value, NodeType.ACTIVATION.value)
    
def test_normalizationTypes():
    """
    Tests the ability to use NormalizationType.
    """	    
    # NO_NORM
    var normalizationType = NormalizationType.NO_NORM
    assert_equal(normalizationType.value, NormalizationType.NO_NORM.value)
        
    # BATCH_NORM
    normalizationType = NormalizationType.BATCH_NORM
    assert_equal(normalizationType.value, NormalizationType.BATCH_NORM.value)
    
def test_poolingTypes():
    """
    Tests the ability to use PoolingType.
    """	    
    # NO_POOLING
    var poolingType = PoolingType.NO_POOLING
    assert_equal(poolingType.value, PoolingType.NO_POOLING.value)
        
    # MAX_POOLING
    poolingType = PoolingType.MAX_POOLING
    assert_equal(poolingType.value, PoolingType.MAX_POOLING.value)
    
def test_activationTypes():
    """
    Tests the ability to use ActivationType.
    """	    
    # RELU
    var activationType = ActivationType.RELU
    assert_equal(activationType.value, ActivationType.RELU.value)
        
    # NONE
    activationType = ActivationType.NONE
    assert_equal(activationType.value, ActivationType.NONE.value)
    
def test_inputNode():
    """
    Tests the ability to construct and use the InputNode struct.
    """
    torch = Python.import_module("torch")

    var node = InputNode(inputShape=torch.Size([4, 3, 32, 32]))
    assert_equal(node.numChannels, 3)
    assert_equal(node.name, 'input')
    assert_equal(node.displayName, 'Input(numChannels=3)')
    
def test_inputNodeWrapper():
    """
    Tests the ability to construct and use the Node struct wrapper for InputNode struct.
    """
    torch = Python.import_module("torch")

    node = Node(name='inputNode', displayName='Input Node')
    node.inputNode = InputNode(inputShape=torch.Size([4, 3, 32, 32]))
    assert_equal(node.inputNode.value().numChannels, 3)
    assert_equal(node.inputNode.value().name, 'input')
    assert_equal(node.inputNode.value().displayName, 'Input(numChannels=3)')
    
def test_OutputNode():
    """
    Tests the ability to construct and use the OutputNode stuct.
    """
    node = OutputNode()
    assert_equal(node.name, 'output')
    assert_equal(node.displayName, 'Output')
    
def test_OutputNodeWrapper():
    """
    Tests the ability to construct and use the Node struct wrapper for OutputNode struct.
    """
    node = Node(name='outputNode', displayName='Output Node')
    node.outputNode = OutputNode()
    assert_equal(node.outputNode.value().name, 'output')
    assert_equal(node.outputNode.value().displayName, 'Output')
    
def test_NormalizationNode():
    """
    Tests the ability to construct and use a node of the NormalizationNode struct.
    """
    uuid = Python.import_module("uuid")
    pytorchLayerId = uuid.uuid4()
    node = NormalizationNode(name="name", 
                             normalizationType=NormalizationType.BATCH_NORM, 
                             numFeatures=12, pytorchLayerId=pytorchLayerId)
    assert_equal(node.name, 'name')
    assert_equal(node.displayName, 'Batch Normalization')
    assert_equal(node.normalizationType.value, NormalizationType.BATCH_NORM.value)
    assert_equal(pytorchLayerId, node.pytorchLayerId)
    
def test_PoolingNode():
    """
    Tests the ability to construct and use a node of the PoolingNode struct.
    """
    node = PoolingNode(name='name', poolingType=PoolingType.MAX_POOLING)
    assert_equal(node.name, 'name')
    assert_equal(node.displayName, 'Max Pooling')
    assert_equal(node.poolingType.value, PoolingType.MAX_POOLING.value)
    
def test_ActivationNode():
    """
    Tests the ability to construct and use a node of the ActivationNode struct.
    """
    node = ActivationNode(name='name', activationType=ActivationType.RELU)
    assert_equal(node.name, 'name')
    assert_equal(node.displayName, 'Relu Activation')
    assert_equal(node.activationType.value, ActivationType.RELU.value)

def test_FlattenNode():
    """
    Tests the ability to construct and use a node of the FlattenNode struct.
    """
    node = FlattenNode(name='name')
    assert_equal(node.name, 'name')
    assert_equal(node.displayName, 'Flatten')

def test_CovolutionalNode():
    """
    Tests the ability to construct and use a node of the CovolutionalNode struct.
    """
    # Valid Construction with int for kernel size
    uuid = Python.import_module("uuid")
    var pytorchLayerId = uuid.uuid4()
    node = ConvolutionalNode(name='name', kernel_size=3, 
                             maxNumInputChannels=128, 
                             maxNumOutputChannels=128, 
                             numOutputChannels=32,
                             layer=0, pytorchLayerId=pytorchLayerId)
    assert_true(node.name == 'name')
    assert_true(node.layer == 0)
    assert_true(node.pytorchLayerId == pytorchLayerId)
    assert_true(node.displayName == '3x3 Conv(oc=32)')
    assert_true(node.kernel_size == 3)
    assert_true(node.maxNumInputChannels == 128)
    assert_true(node.numOutputChannels == 32)

def test_LinearNode():
    """
    Tests the ability to construct and use a node of the LinearNode struct.
    """
    uuid = Python.import_module("uuid")
    pytorchLayerId = uuid.uuid4()
    node = LinearNode(name='name', 
                      maxNumInFeatures=512, 
                      maxNumOutFeatures=512,
                      numOutFeatures=32, 
                      layer=1, pytorchLayerId=pytorchLayerId)
    assert_true(node.name == 'name')
    assert_true(node.layer == 1)
    assert_true(node.pytorchLayerId == pytorchLayerId)
    assert_true(node.maxNumInFeatures == 512)
    assert_true(node.numOutFeatures == 32)
    assert_true(node.displayName == 'Linear(of=32)')
    
def test_Node():
    """
    Tests the ability to construct and use a node of the Node struct.
    """
    uuid = Python.import_module("uuid")
    
    node = Node(name='name', displayName='Node Name')
    
    pytorchLayerId = uuid.uuid4()
    
    #var convNode: Optional[ConvolutionalNode] = None
    #convNode = None
    #convNode = ConvolutionalNode(name='name', kernel_size=3, 
    #                     maxNumInputChannels=128, 
    #                     maxNumOutputChannels=128, 
    #                     numOutputChannels=32,
    #                     layer=0, pytorchLayerId=pytorchLayerId)
    
    pytorchLayerId = uuid.uuid4()
    linearNode = LinearNode(name='name', 
                      maxNumInFeatures=512, 
                      maxNumOutFeatures=512,
                      numOutFeatures=32, 
                      layer=1, pytorchLayerId=pytorchLayerId)
                      
    var genNode: AnyType


def main():
    print('hi')
    from collections import Optional
    var a = Optional(1)
    var b = Optional[Int](None)
    if a:
        print(a.value())  # prints 1
    if b:  # bool(b) is False, so no print
        print(b.value())
    var c = a.or_else(2)
    var d = b.or_else(2)
    print(c)  # prints 1
    print(d)  # prints 2
    
    var node: Optional[ConvolutionalNode]
    node = Optional[ConvolutionalNode](None)
    #node = Node(name='name', displayName='Node Name')

