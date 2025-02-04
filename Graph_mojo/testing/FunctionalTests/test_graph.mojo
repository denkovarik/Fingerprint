from testing import assert_equal, assert_not_equal, assert_true, assert_false
from python import Python, PythonObject
from structs.Nodes import NodeType, NormalizationType, PoolingType, ActivationType
from structs.Graph import Graph


# Functional Tests
    
def test_addNormalizationLayer():
    """
    Tests the method for addin the normalization node to the graph.
    """
    torch = Python.import_module("torch")    
    var grph = Graph()
    grph.addNormalizationLayer(12)
    assert_equal(len(grph.nodes), 2)
    for item in grph.nodes.items():
        assert_true(item[].value.getNodeType().value == NodeType.NORMALIZATION.value)
    
def test_addInputLayer():
    """
    Tests the method for addin the input node to the graph.
    """
    torch = Python.import_module("torch")    
    var grph = Graph()
    grph.addInputLayer(inputShape=torch.Size([4, 3, 32, 32]))
    assert_equal(len(grph.nodes), 1)
    for item in grph.nodes.items():
        assert_true(item[].value.getNodeType().value == NodeType.INPUT.value)
    
def test_addActivationLayer():
    """
    Tests the method for adding the Activation Layer to the graph.
    """ 
    var grph = Graph()
    grph.addActivationLayer()
    assert_equal(len(grph.nodes), 2)
    for item in grph.nodes.items():
        assert_true(item[].value.getNodeType().value == NodeType.ACTIVATION.value)
    
def test_addPoolingLayer():
    """
    Tests the method for adding the Pooling Layer to the graph.
    """   
    var grph = Graph()
    grph.addPoolingLayer()
    assert_equal(len(grph.nodes), 2)   
    for item in grph.nodes.items():
        assert_true(item[].value.getNodeType().value == NodeType.POOLING.value)

    
    
def main():
    torch = Python.import_module("torch")    
    var grph = Graph()
    var inputShape=torch.Size([4, 3, 32, 32])
    var out_shape = grph.addInputLayer(inputShape=inputShape)
    print(out_shape)
    print(out_shape[1])
    grph.addNormalizationLayer(out_shape[1])
    #out_shape = grph.addConvolutionalLayers(inputShape=out_shape)
    grph.addActivationLayer()
    grph.addPoolingLayer()
    
    
    print(len(grph.nodes))
    var nodeLen: Int = len(grph.nodes)
    
    for item in grph.nodes.items():
        print(item[].key)
        edgLen = len(grph.edges[item[].key])
        for i in range(edgLen):
            var edges: List[String] = grph.edges[item[].key]
            edg = edges[i]
            print('\t->' + edg)
        print("")
