from testing import assert_equal, assert_not_equal
from python import Python, PythonObject
from structs.SharedLinear import SharedLinear


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
