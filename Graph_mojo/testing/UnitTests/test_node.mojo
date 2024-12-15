# Content of test_quickstart.mojo
from testing import assert_equal
from python import Python
from classes.Nodes import NodeType


def test_execution():
    # Just tests running a mojo test
    assert_equal(0, 0)

def test_node_types():
    """
    Tests the ability to use NodeType.
    """	
    assert_equal(0, 0)
    
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
    assert_equal(nodeType.value, 'none')
