from testing import assert_equal, assert_not_equal, assert_true, assert_false
from python import Python, PythonObject
from structs.SharedConv2d import SharedConv2d


def test_execution():
    # Just tests running a mojo test
    assert_equal(0, 0)
    assert_true(True)

def test_Construction():
    """
    Tests construction of the SharedConv2DTests class.
    """
    sharedConv = SharedConv2d(kernel_size=3, in_channels=32, out_channels=32)
    #self.assertTrue(isinstance(sharedConv, SharedConv2d))

def test_ForwardPass():
    """
    Tests the forward pass for the SharedConv2DTests class.
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
        
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    
    var weights = torch.nn.init.kaiming_uniform_(torch.empty(8, 3, 3, 3), mode='fan_in', nonlinearity='relu')
    var weightsSub = torch.narrow(weights, 0, 0, 4).clone()

    # Construct control conv2d layer
    var conv2d = nn.Conv2d(3, 4, 3)
    conv2d.weight.data = weightsSub
    conv2d.bias.data.zero_()
    # Construct shared conv2d layer
    var sharedConv2d = SharedConv2d(3, 8, 3)
    sharedConv2d.weight = nn.Parameter(weights)
    sharedConv2d.bias.data.zero_()
    # Make sure initialization was done correctly
    assert_true(conv2d.kernel_size == 3)
    assert_true(sharedConv2d.kernel_size == 3)
    assert_true(torch.allclose(conv2d.weight, weightsSub))
    assert_true(torch.allclose(sharedConv2d.weight, weights))
    assert_true(torch.all(conv2d.bias.eq(0)))
    assert_true(torch.all(sharedConv2d.bias.eq(0)))
    assert_true(torch.allclose(conv2d.weight, torch.narrow(sharedConv2d.weight, 0, 0, 4)))
    # Get test batch
    #testBatchPath = os.path.join(os.path.dirname(os.path.abspath(sys.argv[0])), 'Graph_mojo/testing/UnitTests/TestFiles/cifar10_test_batch_pickle')
    testBatchPath = 'testing/UnitTests/TestFiles/cifar10_test_batch_pickle'
    
    assert_true(os.path.exists(testBatchPath))
    Python.add_to_path(".")
    utils = Python.import_module("utils")
    imgData = utils.unpickle_test_data(testBatchPath, 4)
    batch = imgData.reshape(4, 3, 32, 32)
    tensorData = torch.tensor(batch, dtype=torch.float32)
    # Forward prop
    var outConv2d = conv2d(tensorData)
    outSharedConv2d = sharedConv2d.forward(tensorData, 3, 4)
    # Test the output is the same
    assert_true(torch.allclose(outConv2d, outSharedConv2d))

def test_Print():
    """
    Tests the overloaded to string fucntion.
    """
    torch = Python.import_module("torch")
    nn = Python.import_module("torch.nn")
    conv2dLayer = nn.Conv2d(3, 8, kernel_size=3)
    sharedConv2d = SharedConv2d(in_channels=3, out_channels=8, kernel_size=3)
    exp = "SharedConv2d(3, 8, kernel_size=3)"
    assert_true(sharedConv2d.__str__() == exp)
