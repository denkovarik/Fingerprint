from testing import assert_equal, assert_not_equal, assert_true, assert_false
from python import Python, PythonObject
from structs.Nodes import NodeType
from structs.Nodes import InputNode, FlattenNode, NodeTrait


def test_execution():
    """
    Just tests running a mojo test.
    """
    assert_equal(0, 0)
    
def test_constructionInputNode():
    """
    Tests the ability to construct and use the InputNode struct.
    """
    torch = Python.import_module("torch")

    var node = InputNode(inputShape=torch.Size([4, 3, 32, 32]))
    assert_equal(node.numChannels, 3)
    assert_equal(node.name, 'input')
    assert_equal(node.displayName, 'Input(numChannels=3)')
    
def test_forward():
    """
    Test forward propigation for the FlattenNode struct.
    """
    torch = Python.import_module("torch")
    
    node = InputNode(inputShape=torch.Size([4, 3, 32, 32]))
    
    var inputTensor: PythonObject = torch.randn(1, 3, 5, 5)    
    var nodeTestOutput = node.forward(inputTensor)
    
    assert_equal(nodeTestOutput, inputTensor)
    
def test_toString():
    """
    Tests the to string overloaded function.
    """
    torch = Python.import_module("torch")
    
    node = InputNode(inputShape=torch.Size([4, 3, 32, 32]))
                  
    assert_equal(node.__str__(), 'Input()')
