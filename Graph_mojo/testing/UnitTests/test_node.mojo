# Content of test_quickstart.mojo
from testing import assert_equal, assert_not_equal
from python import Python
from structs.Nodes import NodeType, NormalizationType, PoolingType, ActivationType, InputNode


def test_execution():
    # Just tests running a mojo test
    assert_equal(0, 0)

def test_nodeTypes():
    """
    Tests the ability to use NodeType.
    """	    
    # INPUT
    var nodeType = NodeType('input')
    assert_equal(nodeType.value, 'input')
        
    # OUTPUT
    nodeType = NodeType('output')
    assert_equal(nodeType.value, 'output')
        
    # CONVOLUTION
    nodeType = NodeType('convolution')
    assert_equal(nodeType.value, 'convolution')
        
    # NORMALIZATION
    nodeType = NodeType('normalization')
    assert_equal(nodeType.value, 'normalization')
        
    # POOLING
    nodeType = NodeType('pooling')
    assert_equal(nodeType.value, 'pooling')
        
    # FLATTEN
    nodeType = NodeType('flatten')
    assert_equal(nodeType.value, 'flatten')
        
    # LINEAR
    nodeType = NodeType('linear')
    assert_equal(nodeType.value, 'linear')
            
    # ACTIVATION
    nodeType = NodeType('activation')
    assert_equal(nodeType.value, 'activation')
                
    # Unrecognized Node Type
    nodeType = NodeType('Que')
    assert_not_equal(nodeType.value, 'Que')
    assert_equal(nodeType.value, 'none')
    
def test_normalizationTypes():
    """
    Tests the ability to use NormalizationType.
    """	    
    # NO_NORM
    var normalizationType = NormalizationType('noNorm')
    assert_equal(normalizationType.value, 'noNorm')
        
    # BATCH_NORM
    normalizationType = NormalizationType('batchNorm')
    assert_equal(normalizationType.value, 'batchNorm')

    # Unrecognized Node Type
    normalizationType = NormalizationType('Que')
    assert_not_equal(normalizationType.value, 'Que')
    assert_equal(normalizationType.value, 'none')
    
def test_poolingTypes():
    """
    Tests the ability to use PoolingType.
    """	    
    # NO_POOLING
    var poolingType = PoolingType('noPooling')
    assert_equal(poolingType.value, 'noPooling')
        
    # MAX_POOLING
    poolingType = PoolingType('maxPooling')
    assert_equal(poolingType.value, 'maxPooling')

    # Unrecognized Node Type
    poolingType = PoolingType('Que')
    assert_not_equal(poolingType.value, 'Que')
    assert_equal(poolingType.value, 'none')
    
def test_activationTypes():
    """
    Tests the ability to use ActivationType.
    """	    
    # RELU
    var activationType = ActivationType('reluActivation')
    assert_equal(activationType.value, 'reluActivation')
        
    # NONE
    activationType = ActivationType('none')
    assert_equal(activationType.value, 'none')

    # Unrecognized Node Type
    activationType = ActivationType('Que')
    assert_not_equal(activationType.value, 'Que')
    assert_equal(activationType.value, 'none')
    
def test_inputNode():
    """
    Tests the ability to construct and use the InputNode struct
    """
    nn = Python.import_module("torch.nn")
    np = Python.import_module("numpy")
    torch = Python.import_module("torch")
    
    var size_list = Python.list()
    size_list = [4, 3, 32, 32]
    #var shape = List[Int](4, 3, 32, 32)
    var inputShape = torch.Size([4, 3, 32, 32])
    var node = InputNode()
    #var node = InputNode(inputShape=torch.Size([4, 3, 32, 32]))
    #assert_true(node.numChannels == 3)
    #assert_true(node.name == 'input')
    #assert_true(node.displayName == 'Input(numChannels=3)')