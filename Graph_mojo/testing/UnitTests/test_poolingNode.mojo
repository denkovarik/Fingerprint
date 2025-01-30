# Content of test_quickstart.mojo
from testing import assert_equal, assert_not_equal, assert_true, assert_false
from python import Python, PythonObject
from structs.Nodes import NodeType, PoolingType
from structs.Nodes import, PoolingNode, NodeTrait


def test_execution():
    # Just tests running a mojo test
    assert_equal(0, 0)
    
def test_construction():
    """
    Tests the ability to construct and use a node of the PoolingNode struct.
    """
    uuid = Python.import_module("uuid")
    pytorchLayerId = uuid.uuid4()
    node = PoolingNode(name='name', poolingType=PoolingType.MAX_POOLING)
    assert_equal(node.name, 'name')
    assert_equal(node.displayName, 'Max Pooling')
    assert_equal(node.poolingType.value, PoolingType.MAX_POOLING.value)
    
def test_forwardNoPooling():
    """
    Test forward propigation for the PoolingNode struct with no pooling set.
    """
    torch = Python.import_module("torch")
    nn = Python.import_module("torch.nn")
    uuid = Python.import_module("uuid")
    pytorchLayerId = uuid.uuid4()
    
    noPooling = PoolingNode(name="name", 
                             poolingType=PoolingType.NO_POOLING)
    
    var inputTensor: PythonObject = torch.randn(1, 3, 5, 5)
    
    var nodeNoPoolingTestOutput = noPooling.forward(inputTensor)
    assert_true(torch.allclose(inputTensor, nodeNoPoolingTestOutput))

def test_forwardMaxPooling():
    """
    Test forward propigation for the PoolingNode struct with no pooling set.
    """
    torch = Python.import_module("torch")
    nn = Python.import_module("torch.nn")
    uuid = Python.import_module("uuid")
    pytorchLayerId = uuid.uuid4()

    maxPooling = PoolingNode(name="name", 
                             poolingType=PoolingType.MAX_POOLING)
    
    var inputTensor: PythonObject = torch.randn(1, 3, 8, 8)
    var maxPoolingModule: PythonObject = nn.MaxPool2d(kernel_size=2, stride=2) 
    
    var maxPoolingModuleOuptput = maxPoolingModule(inputTensor)
    var nodeMaxPoolingTestOutput = maxPooling.forward(inputTensor)
    assert_false(inputTensor.shape == nodeMaxPoolingTestOutput.shape)
    assert_true(torch.allclose(maxPoolingModuleOuptput, nodeMaxPoolingTestOutput))
    
def test_toStringNoPooling():
    """
    Tests the to string overloaded function for NO_NORM type.
    """
    uuid = Python.import_module("uuid")
    pytorchLayerId = uuid.uuid4()

    noPooling = PoolingNode(name="name", 
                             poolingType=PoolingType.NO_POOLING)
                  
    assert_equal(noPooling.__str__(), 'NoPooling2d()')

def test_toStringMaxPooling():
    """
    Tests the to string overloaded function for MAX_POOLING type.
    """
    uuid = Python.import_module("uuid")
    pytorchLayerId = uuid.uuid4()
    nn = Python.import_module("torch.nn")

    maxPooling = PoolingNode(name="name", 
                             poolingType=PoolingType.MAX_POOLING)
    var maxPoolingModule: PythonObject = nn.MaxPool2d(kernel_size=2, stride=2) 
                  
    assert_equal(maxPooling.__str__(), str(maxPoolingModule))
    