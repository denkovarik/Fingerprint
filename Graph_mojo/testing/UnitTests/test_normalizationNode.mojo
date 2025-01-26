# Content of test_quickstart.mojo
from testing import assert_equal, assert_not_equal, assert_true, assert_false
from python import Python, PythonObject
from structs.Nodes import NodeType, NormalizationType
from structs.Nodes import NormalizationNode, NodeTrait


def test_execution():
    # Just tests running a mojo test
    assert_equal(0, 0)
    
def test_construction():
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
    
def test_forwardNoNormalization():
    """
    Test forward propigation for the NormalizationNode struct with no normalization set.
    """
    torch = Python.import_module("torch")
    nn = Python.import_module("torch.nn")
    uuid = Python.import_module("uuid")
    pytorchLayerId = uuid.uuid4()
    
    noNorm = NormalizationNode(name="name", 
                             normalizationType=NormalizationType.NO_NORM, 
                             numFeatures=3, pytorchLayerId=pytorchLayerId)
    
    var inputTensor: PythonObject = torch.randn(1, 3, 5, 5)
    
    var nodeNoNormTestOutput = noNorm.forward(inputTensor)
    assert_true(torch.allclose(inputTensor, nodeNoNormTestOutput))

def test_forwardBatchNormalization():
    """
    Test forward propigation for the NormalizationNode struct with no normalization set.
    """
    torch = Python.import_module("torch")
    nn = Python.import_module("torch.nn")
    uuid = Python.import_module("uuid")
    pytorchLayerId = uuid.uuid4()

    batchNorm = NormalizationNode(name="name", 
                             normalizationType=NormalizationType.BATCH_NORM, 
                             numFeatures=3, pytorchLayerId=pytorchLayerId)
    
    var inputTensor: PythonObject = torch.randn(1, 3, 5, 5)
    var batchNormModule: PythonObject = nn.BatchNorm2d(3) 
    
    var batchNormModuleOuptput = batchNormModule(inputTensor)
    var nodeBatchNormTestOutput = batchNorm.forward(inputTensor)
    assert_false(torch.allclose(inputTensor, nodeBatchNormTestOutput))
    assert_true(torch.allclose(batchNormModuleOuptput, nodeBatchNormTestOutput))
    
def test_toStringNoNorm():
    """
    Tests the to string overloaded function for NO_NORM type.
    """
    uuid = Python.import_module("uuid")
    pytorchLayerId = uuid.uuid4()

    var noNorm = NormalizationNode(name="name", 
                         normalizationType=NormalizationType.NO_NORM, 
                         numFeatures=3, pytorchLayerId=pytorchLayerId)
                  
    assert_equal(noNorm.__str__(), 'NoNorm2d()')

def test_toStringBatchNorm():
    """
    Tests the to string overloaded function for NO_NORM type.
    """
    uuid = Python.import_module("uuid")
    pytorchLayerId = uuid.uuid4()
    nn = Python.import_module("torch.nn")

    var batchNorm = NormalizationNode(name="name", 
                         normalizationType=NormalizationType.BATCH_NORM, 
                         numFeatures=3, pytorchLayerId=pytorchLayerId)
    var batchNormModule: PythonObject = nn.BatchNorm2d(3)
                  
    assert_equal(batchNorm.__str__(), str(batchNormModule))
    