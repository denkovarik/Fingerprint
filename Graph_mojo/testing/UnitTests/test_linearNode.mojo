from testing import assert_equal, assert_not_equal, assert_true, assert_false
from python import Python, PythonObject
from structs.Nodes import Node, LinearNode, NodeTrait


def test_execution():
    # Just tests running a mojo test
    assert_equal(0, 0)

def test_constructionLinearNode():
    """
    Tests the ability to construct and use a node of the LinearNode struct.
    """
    uuid = Python.import_module("uuid")
    pytorchLayerId = uuid.uuid4()
    node = LinearNode(name='name', 
                      maxNumInFeatures=32, 
                      maxNumOutFeatures=32,
                      numOutFeatures=16, 
                      layer=1, pytorchLayerId=pytorchLayerId)
    assert_true(node.name == 'name')
    assert_true(node.layer == 1)
    assert_true(node.pytorchLayerId == pytorchLayerId)
    assert_true(node.maxNumInFeatures == 32)
    assert_true(node.numOutFeatures == 16)
    assert_true(node.displayName == 'Linear(of=16)')   

def test_Print():
    """
    Tests the overloaded to string fucntion.
    """
    torch = Python.import_module("torch")
    nn = Python.import_module("torch.nn")
 
    uuid = Python.import_module("uuid")
    pytorchLayerId = uuid.uuid4() 
    linearLayer = nn.Linear(in_features=10, out_features=5)
    node = LinearNode(name='name', 
                  maxNumInFeatures=32, 
                  maxNumOutFeatures=32,
                  numOutFeatures=16, 
                  layer=1, pytorchLayerId=pytorchLayerId)
    exp = 'Linear(of=16)'
    assert_equal(node.__str__(), exp)
    
def test_forward():
    """
    Tests just calling the LinearNode struct forward function.
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
    
    var random_tensor = torch.rand(4000, 4000)
    
    var random_tensor_clone = random_tensor.clone()
    assert_true(torch.allclose(random_tensor, random_tensor_clone)) 
    
    var random_tensor2 = torch.rand(4000, 4000)
    
    random_tensor2_sub = torch.narrow(torch.narrow(random_tensor2, 0, 0, 8), 1, 0, 3072).clone()
    assert_true(torch.allclose(torch.narrow(torch.narrow(random_tensor2, 0, 0, 8), 1, 0, 3072), random_tensor2_sub)) 
    
    random_tensor = torch.rand(4000, 4000)
    
    var random_tensor_defective = torch.rand(4000, 4000)
    assert_false(torch.allclose(random_tensor, random_tensor_defective)) 
    
    var fc1 = nn.Linear(3072, 8)
    fc1.weight.data = weightsSub
    fc1.bias.data.zero_()
    var fc1_out = fc1(flattened_tensor)
    
    uuid = Python.import_module("uuid")
    pytorchLayerId = uuid.uuid4() 
    node = LinearNode(name='name', 
              maxNumInFeatures=4000, 
              maxNumOutFeatures=4000,
              numOutFeatures=8, 
              layer=1, pytorchLayerId=pytorchLayerId)
    node.pytorchLayer.weight = nn.Parameter(weights)
    node.pytorchLayer.bias.data.zero_() 
    var shared_out = node.forward(flattened_tensor)
    assert_true(torch.allclose(fc1_out, shared_out)) 
    