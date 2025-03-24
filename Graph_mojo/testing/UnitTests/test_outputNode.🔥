from testing import assert_equal, assert_not_equal, assert_true, assert_false
from python import Python, PythonObject
from structs.Nodes import NodeType
from structs.Nodes import OutputNode, NodeTrait


def test_execution():
    """
    Just tests running a mojo test.
    """
    assert_equal(0, 0)
    
def test_constructionOutputNode():
    """
    Tests the ability to construct and use the OutputNode stuct.
    """
    node = OutputNode()
    assert_equal(node.name, 'output')
    assert_equal(node.displayName, 'Output')
    
def test_forwardOutputNode():
    """
    Test forward propigation for the OutputNode struct.
    """
    torch = Python.import_module("torch")
    
    node = OutputNode()
    
    var inputTensor: PythonObject = torch.randn(1, 3, 5, 5)    
    var nodeTestOutput = node.forward(inputTensor)
    
    assert_equal(nodeTestOutput, inputTensor)
    
def test_toStringOutputNode():
    """
    Tests the to string overloaded function.
    """    
    node = OutputNode()           
    assert_equal(node.__str__(), 'Output()')
