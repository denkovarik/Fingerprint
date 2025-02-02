from testing import assert_equal, assert_not_equal, assert_true, assert_false
from python import Python, PythonObject
from structs.Nodes import NodeType
from structs.Nodes import FlattenNode, NodeTrait


def test_execution():
    """
    Just tests running a mojo test.
    """
    assert_equal(0, 0)
    
def test_construction():
    """
    Tests the ability to construct and use a node of the FlattenNode struct.
    """
    node = FlattenNode(name='name')
    assert_equal(node.name, 'name')
    assert_equal(node.displayName, 'Flatten')
    
def test_forward():
    """
    Test forward propigation for the FlattenNode struct.
    """
    torch = Python.import_module("torch")
    nn = Python.import_module("torch.nn")
    
    node = FlattenNode(name='name')
    flattenLayer = nn.Flatten()
    
    var inputTensor: PythonObject = torch.randn(1, 3, 5, 5)    
    var nodeFlattenTestOutput = node.forward(inputTensor)
    var flattenLayerControlOut = flattenLayer(inputTensor)
    
    assert_not_equal(inputTensor.shape, nodeFlattenTestOutput.shape)
    assert_equal(nodeFlattenTestOutput.shape, flattenLayerControlOut.shape)
    
def test_toString():
    """
    Tests the to string overloaded function.
    """
    nn = Python.import_module("torch.nn")

    node = FlattenNode(name='name')
    flattenLayer = nn.Flatten()
                  
    assert_equal(node.__str__(), str(flattenLayer))
