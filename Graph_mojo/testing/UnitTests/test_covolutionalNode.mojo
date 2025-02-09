from testing import assert_equal, assert_not_equal, assert_true, assert_false
from python import Python, PythonObject
from structs.Nodes import Node, ConvolutionalNode, NodeTrait


def test_execution():
    """
    Just tests running a mojo test
    """
    assert_equal(0, 0)
    assert_true(True)

def test_constructionCovolutionalNode():
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
    

def test_forwardPassCovolutionalNode():
    """
    Tests the forward pass for the ConvolutionalNode class.
    """
    torch = Python.import_module("torch")
    nn = Python.import_module("torch.nn")
    init = Python.import_module("torch.nn.init")
    F = Python.import_module("torch.nn.functional")
    np = Python.import_module("numpy")
    math = Python.import_module("math")
    random = Python.import_module("random")
    os = Python.import_module("os")
    pickle = Python.import_module("pickle")
    sys = Python.import_module("sys")
    
    # Get test batch
    testBatchPath = 'testing/UnitTests/TestFiles/cifar10_test_batch_pickle'
    assert_true(os.path.exists(testBatchPath))
    Python.add_to_path(".")
    utils = Python.import_module("utils")
    imgData = utils.unpickle_test_data(testBatchPath, 4)
    batch = imgData.reshape(4, 3, 32, 32)
    tensorData = torch.tensor(batch, dtype=torch.float32)
        
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    
    var weights = torch.nn.init.kaiming_uniform_(torch.empty(16, 6, 3, 3), mode='fan_in', nonlinearity='relu')
    var weightsSub = torch.narrow(torch.narrow(weights, 0, 0, 8), 1, 0, 3).clone()
    
    var random_tensor = torch.rand(8, 3, 3, 3)
    
    var random_tensor_clone = random_tensor.clone()
    assert_true(torch.allclose(random_tensor, random_tensor_clone)) 
    
    var random_tensor2 = torch.rand(16, 6, 3, 3)
    
    random_tensor2_sub = torch.narrow(torch.narrow(random_tensor2, 0, 0, 8), 1, 0, 3).clone()
    assert_true(torch.allclose(torch.narrow(torch.narrow(random_tensor2, 0, 0, 8), 1, 0, 3), random_tensor2_sub)) 
    
    random_tensor = torch.rand(8, 3, 3, 3)

    var random_tensor_defective = torch.rand(8, 3, 3, 3)
    assert_false(torch.allclose(random_tensor, random_tensor_defective)) 
    
    var conv2d = nn.Conv2d(3, 8, 3)
    conv2d.weight.data = weightsSub
    conv2d.bias.data.zero_()
    assert_true(conv2d.kernel_size == 3)
    assert_true(torch.allclose(conv2d.weight, weightsSub))
    assert_true(torch.all(conv2d.bias.eq(0)))
    var outConv2d = conv2d(tensorData)

    uuid = Python.import_module("uuid")
    var pytorchLayerId = uuid.uuid4()
    node = ConvolutionalNode(name='name', kernel_size=3, 
                             maxNumInputChannels=6, 
                             maxNumOutputChannels=16, 
                             numOutputChannels=8,
                             layer=0, pytorchLayerId=pytorchLayerId)
                             
    node.pytorchLayer.weight = nn.Parameter(weights)
    node.pytorchLayer.bias.data.zero_()

    assert_true(node.kernel_size == 3)
    assert_true(torch.allclose(node.pytorchLayer.weight, weights))
    assert_true(torch.all(node.pytorchLayer.bias.eq(0)))   
    assert_true(torch.allclose(conv2d.weight, torch.narrow(torch.narrow(weights, 0, 0, 8), 1, 0, 3)))
    node.initSubWeights(tensorData, 3, 8)
    outSharedConv2d = node.forward(tensorData)
    assert_true(torch.allclose(outConv2d, outSharedConv2d)) 

def test_printCovolutionalNode():
    """
    Tests the overloaded to string fucntion.
    """
    torch = Python.import_module("torch")
    nn = Python.import_module("torch.nn")
    uuid = Python.import_module("uuid")
    var pytorchLayerId = uuid.uuid4()
    node = ConvolutionalNode(name='name', kernel_size=3, 
                             maxNumInputChannels=3, 
                             maxNumOutputChannels=8, 
                             numOutputChannels=8,
                             layer=0, pytorchLayerId=pytorchLayerId)
    exp = "SharedConv2d(3, 8, kernel_size=3)"
    assert_true(node.__str__() == exp)
    