# Content of test_quickstart.mojo
from testing import assert_equal, assert_not_equal, assert_true, assert_false
from python import Python, PythonObject
from structs.Nodes import NodeType, PoolingType, ActivationType
from structs.Nodes import ActivationNode, PoolingNode, NodeTrait


def test_execution():
    # Just tests running a mojo test
    assert_equal(0, 0)
    
def test_ActivationNode():
    """
    Tests the ability to construct and use a node of the ActivationNode struct.
    """
    node = ActivationNode(name='name', activationType=ActivationType.RELU)
    assert_equal(node.name, 'name')
    assert_equal(node.displayName, 'Relu Activation')
    assert_equal(node.activationType.value, ActivationType.RELU.value)
    
def test_forwardLinearActivation():
    """
    Test forward propigation for the ActivationNode struct with linear activation set.
    """
    torch = Python.import_module("torch")
    nn = Python.import_module("torch.nn")
    uuid = Python.import_module("uuid")
    pytorchLayerId = uuid.uuid4()
    
    linearActivation = ActivationNode(name='name', activationType=ActivationType.LINEAR)
    
    var inputTensor: PythonObject = torch.randn(1, 3, 5, 5)
    
    var linearActivationTestOutput = linearActivation.forward(inputTensor)
    assert_true(torch.allclose(inputTensor, linearActivationTestOutput))

def test_forwardReluActivation():
    """
    Test forward propigation for the ActivationNode struct with relu activation set.
    """
    torch = Python.import_module("torch")
    nn = Python.import_module("torch.nn")
    uuid = Python.import_module("uuid")
    pytorchLayerId = uuid.uuid4()

    reluActivation = ActivationNode(name='name', activationType=ActivationType.RELU)
    
    var inputTensor: PythonObject = torch.randn(1, 3, 8, 8)
    var reluModule: PythonObject = nn.ReLU()
    
    var reluModuleOuptput = reluModule(inputTensor)
    var nodeReluActivationTestOutput = reluActivation.forward(inputTensor)
    assert_false(torch.allclose(inputTensor, nodeReluActivationTestOutput))
    assert_true(torch.allclose(reluModuleOuptput, nodeReluActivationTestOutput))
    
def test_toStringLinearActivation():
    """
    Tests the to string overloaded function for LINEAR activation type.
    """
    uuid = Python.import_module("uuid")
    pytorchLayerId = uuid.uuid4()

    linearActivation = ActivationNode(name='name', activationType=ActivationType.LINEAR)
                  
    assert_equal(linearActivation.__str__(), 'LinearActivation()')

def test_toStringReluActivation():
    """
    Tests the to string overloaded function for RELU type.
    """
    uuid = Python.import_module("uuid")
    pytorchLayerId = uuid.uuid4()
    nn = Python.import_module("torch.nn")

    reluActivation = ActivationNode(name='name', activationType=ActivationType.RELU)
    var reluModule: PythonObject = nn.ReLU()
                  
    assert_equal(reluActivation.__str__(), str(reluModule))
    