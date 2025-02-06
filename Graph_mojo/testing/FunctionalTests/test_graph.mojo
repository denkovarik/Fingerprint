from testing import assert_equal, assert_not_equal, assert_true, assert_false
from python import Python, PythonObject
from structs.Nodes import NodeType, NormalizationType, PoolingType, ActivationType
from structs.Graph import GraphHandler


# Functional Tests
    
def test_addNormalizationLayer():
    """
    Tests the method for adding the normalization node to the graph.
    """
    torch = Python.import_module("torch")    
    var grph = GraphHandler()
    grph.addNormalizationLayer(12)
    assert_equal(len(grph.graph.nodes), 2)
    for item in grph.graph.nodes.items():
        assert_true(item[].value.getNodeType().value == NodeType.NORMALIZATION.value)
    
def test_addInputLayer():
    """
    Tests the method for addin the input node to the graph.
    """
    torch = Python.import_module("torch")    
    var grph = GraphHandler()
    grph.addInputLayer(inputShape=torch.Size([4, 3, 32, 32]))
    assert_equal(len(grph.graph.nodes), 3)
    
def test_addActivationLayer():
    """
    Tests the method for adding the Activation Layer to the graph.
    """ 
    var grph = GraphHandler()
    grph.addActivationLayer()
    assert_equal(len(grph.graph.nodes), 2)
    for item in grph.graph.nodes.items():
        assert_true(item[].value.getNodeType().value == NodeType.ACTIVATION.value)
    
def test_addPoolingLayer():
    """
    Tests the method for adding the Pooling Layer to the graph.
    """   
    var grph = GraphHandler()
    grph.addPoolingLayer()
    assert_equal(len(grph.graph.nodes), 2)   
    for item in grph.graph.nodes.items():
        assert_true(item[].value.getNodeType().value == NodeType.POOLING.value)
        
def test_addConvolutionalLayer():
    """
    Tests the method for adding the CONVOLUTION Layer to the graph.
    """   
    torch = Python.import_module("torch")  
    var grph = GraphHandler()
    var inputShape=torch.Size([4, 3, 32, 32])
    grph.addConvolutionalLayer(layer=0, inputShape=inputShape)
    assert_equal(len(grph.graph.nodes), 8)   
    for item in grph.graph.nodes.items():
        assert_true(item[].value.getNodeType().value == NodeType.CONVOLUTION.value)
        
def test_addFlattenLayer():
    """
    Tests the method for adding the Flatten Layer to the graph.
    """   
    torch = Python.import_module("torch")  
    var grph = GraphHandler()
    var inputShape=torch.Size([4, 3, 32, 32])
    grph.addFlattenLayer(inputShape=inputShape)
    assert_equal(len(grph.graph.nodes), 1)   
    for item in grph.graph.nodes.items():
        assert_true(item[].value.getNodeType().value == NodeType.FLATTEN.value)
        
def test_addLinearLayer():
    """
    Tests the method for adding the Linear Layer to the graph.
    """   
    torch = Python.import_module("torch")  
    var grph = GraphHandler()
    var inputShape=torch.Size([4, 3, 32, 32])
    grph.addLinearLayer(layer=0, inputShape=inputShape)
    assert_equal(len(grph.graph.nodes), 5)   
    for item in grph.graph.nodes.items():
        assert_true(item[].value.getNodeType().value == NodeType.LINEAR.value)
        
def test_addOutputLayer():
    """
    Tests the method for adding the Output Layer to the graph.
    """   
    torch = Python.import_module("torch")  
    var grph = GraphHandler()
    var inputShape=torch.Size([4, 3, 32, 32])
    grph.addOutputLayer()
    assert_equal(len(grph.graph.nodes), 1)   
    for item in grph.graph.nodes.items():
        assert_true(item[].value.getNodeType().value == NodeType.OUTPUT.value)
        


from collections import Set
    
def main():
    torch = Python.import_module("torch")    
    var grph = GraphHandler()
    var inputShape=torch.Size([4, 3, 32, 32])
    
    grph.construct(inputShape)
    #var out_shape = grph.addInputLayer(inputShape=inputShape)
    #out_shape = grph.addConvolutionalLayers(inputShape=out_shape)
    #outShape = grph.addFlattenLayer(out_shape)
    #grph.addLinearLayers(inputShape=outShape)
    #grph.addOutputLayer()
    
    print(len(grph.graph.nodes))
    var nodeLen: Int = len(grph.graph.nodes)
    
    for item in grph.graph.nodes.items():
        print(item[].key)
        edgLen = len(grph.graph.edges[item[].key])
        for i in range(edgLen):
            var edges: List[String] = grph.graph.edges[item[].key]
            edg = edges[i]
            print('\t->' + edg)
        print("")
