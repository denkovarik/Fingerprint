from testing import assert_equal, assert_not_equal, assert_true, assert_false
from python import Python, PythonObject
from collections import Optional
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
    
    # Template
    var random_tensor = torch.rand(8, 3, 3, 3)
    
    # Response
    var random_tensor_clone = random_tensor.clone()
    #assert_true(torch.allclose(random_tensor, random_tensor_clone)) # Hello
    
    # Template
    var random_tensor2 = torch.rand(16, 6, 3, 3)
    
    # Response
    random_tensor2_sub = torch.narrow(torch.narrow(random_tensor2, 0, 0, 8), 1, 0, 3).clone()
    assert_true(torch.allclose(torch.narrow(torch.narrow(random_tensor2, 0, 0, 8), 1, 0, 3), random_tensor2_sub)) # Hello
    
    # Template
    random_tensor = torch.rand(8, 3, 3, 3)
    
    # Response
    var random_tensor_defective = torch.rand(8, 3, 3, 3)
    assert_false(torch.allclose(random_tensor, random_tensor_defective)) # I'm not defective! 
    
    # Template
    var conv2d = nn.Conv2d(3, 8, 3)
    conv2d.weight.data = weightsSub
    conv2d.bias.data.zero_()
    assert_true(conv2d.kernel_size == 3)
    assert_true(torch.allclose(conv2d.weight, weightsSub))
    assert_true(torch.all(conv2d.bias.eq(0)))
    var outConv2d = conv2d(tensorData)
    
    # Response
    var sharedConv2d = SharedConv2d(6, 16, 3)
    sharedConv2d.initSubWeights(tensorData, 3, 8)
    sharedConv2d.weight = nn.Parameter(weights)
    sharedConv2d.bias.data.zero_()    
    assert_true(sharedConv2d.kernel_size == 3)
    assert_true(torch.allclose(sharedConv2d.weight, weights))
    assert_true(torch.all(sharedConv2d.bias.eq(0)))   
    assert_true(torch.allclose(conv2d.weight, torch.narrow(torch.narrow(weights, 0, 0, 8), 1, 0, 3)))
    sharedConv2d.initSubWeights(tensorData, 3, 8)
    outSharedConv2d = sharedConv2d.forward(tensorData)
    assert_true(torch.allclose(outConv2d, outSharedConv2d))   

def test_ForwardPassGPU():
    """
    Tests the forward pass on GPU for the SharedConv2DTests class.
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
         
    # Response
    var sharedConv2d = SharedConv2d(6, 16, 3) 
      
    var device: PythonObject = torch.device("cpu")
    var cuda_available = torch.cuda.is_available()
    if cuda_available:
        device = torch.device("cuda") 

    sharedConv2d.to(device=device)
    tensorData = tensorData.to(device)
    
    sharedConv2d.initSubWeights(tensorData, 3, 8)
    for i in range(100000):
        outSharedConv2d = sharedConv2d.forward(tensorData)

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
    
def test_CalOutputSize():
    """
    Tests method calcOutSize of the SharedConv2DTests class.
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
    
    # Test 1: Kernel Size 3x3
    var conv2d = nn.Conv2d(3, 8,kernel_size=3)
    var sharedConv2d = SharedConv2d(3, 8, kernel_size=3)
    # Calculate what the output size should be
    var calcOutputSize = SharedConv2d.calcOutSize(tensorData.shape, outChannels=8, kernel_size=3, stride=1, padding=0, dilation=1)
    # Forward prop
    var outConv2d = conv2d(tensorData)
    sharedConv2d.initSubWeights(tensorData, 3, 8)
    var outSharedConv2d = sharedConv2d.forward(tensorData)
    assert_true(calcOutputSize == outConv2d.shape)
    assert_true(calcOutputSize == outSharedConv2d.shape)

    # Test 2: Kernel Size 5x5
    conv2d = nn.Conv2d(3, 16, kernel_size=5)
    sharedConv2d = SharedConv2d(5, 16, kernel_size=5)
    # Calculate what the output size should be
    calcOutputSize = SharedConv2d.calcOutSize(tensorData.shape, outChannels=16, kernel_size=5)
    # Forward prop
    outConv2d = conv2d(tensorData)
    sharedConv2d.initSubWeights(tensorData, 3, 16)
    outSharedConv2d = sharedConv2d.forward(tensorData)
    assert_true(calcOutputSize == outConv2d.shape)
    assert_true(calcOutputSize == outSharedConv2d.shape)

    # Test 3: Shared Conv2d ouput number of Channels less than max Kernel Size 3x3
    conv2d = nn.Conv2d(3, 8,kernel_size=3)
    sharedConv2d = SharedConv2d(3, 16, kernel_size=3)
    # Calculate what the output size should be
    calcOutputSize = SharedConv2d.calcOutSize(tensorData.shape, outChannels=8, kernel_size=3)
    # Forward prop
    outConv2d = conv2d(tensorData)
    sharedConv2d.initSubWeights(tensorData, 3, 8)
    outSharedConv2d = sharedConv2d.forward(tensorData)
    assert_true(calcOutputSize == outConv2d.shape)
    assert_true(calcOutputSize == outSharedConv2d.shape)

    # Test 5: Calculating tensor dimensions for multiple runs
    # Conv2d
    var conv2d1 = nn.Conv2d(3, 8, kernel_size=3)
    var conv2d2 = nn.Conv2d(8, 16, kernel_size=5)
    var conv2d3 = nn.Conv2d(16, 32, kernel_size=5)
    # Forward prop Conv2d Layers
    var outConv2d1 = conv2d1(tensorData)
    var outConv2d2 = conv2d2(outConv2d1)
    var outConv2d3 = conv2d3(outConv2d2)
    # SharedConv2d
    var sharedConv2d1 = SharedConv2d(3, 256, kernel_size=3) 
    var sharedConv2d2 = SharedConv2d(256, 256, kernel_size=5) 
    var sharedConv2d3 = SharedConv2d(256, 256, kernel_size=5)
    # Calculated SharedConv2d Shapes
    var calcOutShape1 = SharedConv2d.calcOutSize(tensorData.shape, outChannels=8, kernel_size=3)
    var calcOutShape2 = SharedConv2d.calcOutSize(calcOutShape1, outChannels=16, kernel_size=5)
    var calcOutShape3 = SharedConv2d.calcOutSize(calcOutShape2, outChannels=32, kernel_size=5)
    # Forward Prop Shared Conv2d Layers
    sharedConv2d1.initSubWeights(tensorData, 3, 8)
    var outSharedConv2d1 = sharedConv2d1.forward(tensorData)
    
    sharedConv2d2.initSubWeights(outSharedConv2d1, 8, 16)
    var outSharedConv2d2 = sharedConv2d2.forward(outSharedConv2d1)
        
    sharedConv2d3.initSubWeights(outSharedConv2d2, 16, 32)
    var outSharedConv2d3 = sharedConv2d3.forward(outSharedConv2d2)
    
    # Validate Shapes
    assert_true(calcOutShape1 == outConv2d1.shape)
    assert_true(calcOutShape1 == outSharedConv2d1.shape)
    assert_true(calcOutShape2 == outConv2d2.shape)
    assert_true(calcOutShape2 == outSharedConv2d2.shape)
    assert_true(calcOutShape3 == outConv2d3.shape)
    assert_true(calcOutShape3 == outSharedConv2d3.shape)
    
def test_GetOutputSize():
    """
    Tests method getOutSize of the SharedConv2DTests class.
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
    
    # Test 1: Kernel Size 3x3
    var conv2d = nn.Conv2d(3, 8,kernel_size=3)
    var sharedConv2d = SharedConv2d(3, 8, kernel_size=3)
    # Calculate what the output size should be
    var calcOutputSize = sharedConv2d.getOutSize(tensorData.shape, outChannels=8)
    # Forward prop
    var outConv2d = conv2d(tensorData)
    sharedConv2d.initSubWeights(tensorData, 3, 8)
    var outSharedConv2d = sharedConv2d.forward(tensorData)
    assert_true(calcOutputSize == outConv2d.shape)
    assert_true(calcOutputSize == outSharedConv2d.shape)

    # Test 2: Kernel Size 5x5
    conv2d = nn.Conv2d(3, 16, kernel_size=5)
    sharedConv2d = SharedConv2d(3, 16, kernel_size=5)
    # Calculate what the output size should be
    calcOutputSize = sharedConv2d.getOutSize(tensorData.shape, outChannels=16)
    # Forward prop
    outConv2d = conv2d(tensorData)
    sharedConv2d.initSubWeights(tensorData, 3, 16)
    outSharedConv2d = sharedConv2d.forward(tensorData)
    assert_true(calcOutputSize == outConv2d.shape)
    assert_true(calcOutputSize == outSharedConv2d.shape)

    # Test 3: Shared Conv2d ouput number of Channels less than max 
    #         Kernel Size 3x3
    conv2d = nn.Conv2d(3, 8,kernel_size=3)
    sharedConv2d = SharedConv2d(3, 16, kernel_size=3)
    # Calculate what the output size should be
    calcOutputSize = sharedConv2d.getOutSize(tensorData.shape, outChannels=8)
    # Forward prop
    outConv2d = conv2d(tensorData)
    sharedConv2d.initSubWeights(tensorData, 3, 8)
    outSharedConv2d = sharedConv2d.forward(tensorData)
    assert_true(calcOutputSize == outConv2d.shape)
    assert_true(calcOutputSize == outSharedConv2d.shape)

    # Test 5: Calculating tensor dimensions for multiple runs
    # Conv2d
    var conv2d1 = nn.Conv2d(3, 8, kernel_size=3)
    var conv2d2 = nn.Conv2d(8, 16, kernel_size=5)
    var conv2d3 = nn.Conv2d(16, 32, kernel_size=5)
    # Forward prop Conv2d Layers
    var outConv2d1 = conv2d1(tensorData)
    var outConv2d2 = conv2d2(outConv2d1)
    var outConv2d3 = conv2d3(outConv2d2)
    # SharedConv2d
    var sharedConv2d1 = SharedConv2d(3, 256, kernel_size=3) 
    var sharedConv2d2 = SharedConv2d(256, 256, kernel_size=5) 
    var sharedConv2d3 = SharedConv2d(256, 256, kernel_size=5)
    # Calculated SharedConv2d Shapes
    var calcOutShape1 = sharedConv2d1.getOutSize(tensorData.shape, outChannels=8)
    var calcOutShape2 = sharedConv2d2.getOutSize(calcOutShape1, outChannels=16)
    var calcOutShape3 = sharedConv2d3.getOutSize(calcOutShape2, outChannels=32)
    # Forward Prop Shared Conv2d Layers
    sharedConv2d1.initSubWeights(tensorData, 3, 8)
    var outSharedConv2d1 = sharedConv2d1.forward(tensorData)
        
    sharedConv2d2.initSubWeights(outSharedConv2d1, 8, 16)
    var outSharedConv2d2 = sharedConv2d2.forward(outSharedConv2d1)
        
    sharedConv2d3.initSubWeights(outSharedConv2d2, 16, 32)
    var outSharedConv2d3 = sharedConv2d3.forward(outSharedConv2d2)
    # Validate Shapes
    assert_true(calcOutShape1 == outConv2d1.shape)
    assert_true(calcOutShape1 == outSharedConv2d1.shape)
    assert_true(calcOutShape2 == outConv2d2.shape)
    assert_true(calcOutShape2 == outSharedConv2d2.shape)
    assert_true(calcOutShape3 == outConv2d3.shape)
    assert_true(calcOutShape3 == outSharedConv2d3.shape)
