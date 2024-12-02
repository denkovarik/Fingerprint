# Content of test_quickstart.mojo
from testing import assert_equal
from python import Python



def test_execution():
    # Just tests running a mojo test
    assert_equal(0, 0)

def test_Pytorch_installed():
    torch = Python.import_module("torch")
    nn = Python.import_module("torch.nn")
    np = Python.import_module("numpy")
    # Setup Linear Layer
    linearLayer = nn.Linear(in_features=5, out_features=2)
    weights = torch.stack((torch.tensor([0.1788, 0.3492, -0.2402, -0.2631, 0.2751]), 
        torch.tensor([0.0930, 0.0822, 0.3475, 0.1840, 0.1201])))
    linearLayer.weight = nn.Parameter(weights)
    # Make sure to also set the bias to zeros to match expected output
    linearLayer.bias.data.zero_()
    
    # Setup input and expected output
    inputTensor = torch.empty(1, 5)
    expOut = torch.empty(1, 2)

    # Fill the tensor using FloatTensor
    data = [-2.0942, -0.8275, 0.2748, 0.6571, 2.0056]
    inputTensor[0] = torch.FloatTensor(data)
    
    data2 = [-0.3506,  0.1945]
    expOut[0] = torch.FloatTensor(data2)
    
    # Run Test
    out = linearLayer(inputTensor)
    
def main():
    test_Pytorch_installed()
