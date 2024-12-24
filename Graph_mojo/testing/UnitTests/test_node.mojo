# Content of test_quickstart.mojo
from testing import assert_equal, assert_not_equal
from python import Python, PythonObject
from structs.Nodes import NodeType, NormalizationType, PoolingType, ActivationType, InputNode, OutputNode, NormalizationNode


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
    Tests the ability to construct and use the InputNode struct
    """
    torch = Python.import_module("torch")

    var node = InputNode(inputShape=torch.Size([4, 3, 32, 32]))
    assert_equal(node.numChannels, 3)
    assert_equal(node.name, 'input')
    assert_equal(node.displayName, 'Input(numChannels=3)')
    
def test_OutputNode():
    """
    Tests the ability to construct and use the OutputNode stuct
    """
    node = OutputNode()
    assert_equal(node.name, 'output')
    assert_equal(node.displayName, 'Output')
    
def test_NormalizationNode():
    """
    Tests the ability to construct and use a node of the NormalizationNode 
    struct
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
    