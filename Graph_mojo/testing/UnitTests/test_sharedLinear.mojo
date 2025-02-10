from testing import assert_equal, assert_not_equal, assert_true, assert_false
from python import Python, PythonObject
from structs.SharedLinear import SharedLinear
from memory import UnsafePointer


def test_execution():
    # Just tests running a mojo test
    assert_equal(0, 0)

def test_Construction():
    """
    Tests construction of the SharedLinear struct.
    """
    sharedLinear = SharedLinear(max_in_features=32, max_out_features=32)
    assert_equal(sharedLinear.maxInFeatures, 32)

def test_Print():
    """
    Tests the overloaded to string fucntion.
    """
    torch = Python.import_module("torch")
    nn = Python.import_module("torch.nn")
        
    linearLayer = nn.Linear(in_features=10, out_features=5)
    sharedLinear = SharedLinear(max_in_features=32, max_out_features=32)
    exp = "SharedLinear(max_in_features=32, max_out_features=32)"
    assert_equal(sharedLinear.__str__(), exp)
    
def test_forward():
    """
    Tests just calling the SharedLinear struct forward function.
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
    var flattened_tensor = tensorData.view(4, -1)  # This reshapes it to shape [4, 3072]
    
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    
    var weights = torch.nn.init.kaiming_uniform_(torch.empty(4000, 4000), mode='fan_in', nonlinearity='relu')
    var weightsSub = torch.narrow(torch.narrow(weights, 0, 0, 8), 1, 0, 3072).clone()
    
    # Template
    var random_tensor = torch.rand(4000, 4000)
    
    # Response
    var random_tensor_clone = random_tensor.clone()
    assert_true(torch.allclose(random_tensor, random_tensor_clone)) # Hello
    
    # Template
    var random_tensor2 = torch.rand(4000, 4000)
    
    # Response
    random_tensor2_sub = torch.narrow(torch.narrow(random_tensor2, 0, 0, 8), 1, 0, 3072).clone()
    assert_true(torch.allclose(torch.narrow(torch.narrow(random_tensor2, 0, 0, 8), 1, 0, 3072), random_tensor2_sub)) # Hello
    
    # Template
    random_tensor = torch.rand(4000, 4000)
    
    # Response
    var random_tensor_defective = torch.rand(4000, 4000)
    assert_false(torch.allclose(random_tensor, random_tensor_defective)) # Let's Do This
    
    # Template
    var fc1 = nn.Linear(3072, 8)
    fc1.weight.data = weightsSub
    fc1.bias.data.zero_()
    var fc1_out = fc1(flattened_tensor)
    
    # Response   
    var sharedLinear = SharedLinear(4000, 4000)
    sharedLinear.weight = nn.Parameter(weights)
    sharedLinear.bias.data.zero_() 
    sharedLinear.initSubWeights(flattened_tensor, 3072, 8)
    var shared_out = sharedLinear.forward(flattened_tensor)
    assert_true(torch.allclose(fc1_out, shared_out)) 
    
def test_forwardGPU():
    """
    Tests just calling the SharedLinear struct forward function.
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
    var flattened_tensor = tensorData.view(4, -1)  # This reshapes it to shape [4, 3072]
    
    var device: PythonObject = torch.device("cpu")
    var cuda_available = torch.cuda.is_available()
    if cuda_available:
        device = torch.device("cuda") 

    var sharedLinear = SharedLinear(4000, 4000)
    sharedLinear.to(device=device)
    flattened_tensor = flattened_tensor.to(device)
    sharedLinear.initSubWeights(flattened_tensor, 3072, 8)
    
    for i in range(100000):
        var shared_out = sharedLinear.forward(flattened_tensor)
    
        