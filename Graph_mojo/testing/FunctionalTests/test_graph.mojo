from testing import assert_equal, assert_not_equal, assert_true, assert_false
from python import Python, PythonObject
from structs.Nodes import NodeType, NormalizationType, PoolingType, ActivationType
from structs.Graph import GraphHandler, Graph


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
        assert_true(item[].value[].getNodeType().value == NodeType.NORMALIZATION.value)
    
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
        assert_true(item[].value[].getNodeType().value == NodeType.ACTIVATION.value)
    
def test_addPoolingLayer():
    """
    Tests the method for adding the Pooling Layer to the graph.
    """   
    var grph = GraphHandler()
    grph.addPoolingLayer()
    assert_equal(len(grph.graph.nodes), 2)   
    for item in grph.graph.nodes.items():
        assert_true(item[].value[].getNodeType().value == NodeType.POOLING.value)
        
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
        assert_true(item[].value[].getNodeType().value == NodeType.CONVOLUTION.value)
        
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
        assert_true(item[].value[].getNodeType().value == NodeType.FLATTEN.value)
        
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
        assert_true(item[].value[].getNodeType().value == NodeType.LINEAR.value)
        
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
        assert_true(item[].value[].getNodeType().value == NodeType.OUTPUT.value)
        
def test_sampleArchitecture():
    """
    Tests the method sampleArchitecture().
    """   
    torch = Python.import_module("torch")  
    var grph = GraphHandler()
    var inputShape=torch.Size([4, 3, 32, 32])
    grph.construct(inputShape)
    
    var sample: List[Int] = List[Int](1, 3, 1, 1, 0, 2, 1, 1, 1, 0, 1, 1, 0, 0, 0, 0, 0)
    var sampleGraph: Graph = grph.sampleArchitecture(sample)
    
    var keys: List[String] = List[String]()
    for item in sampleGraph.nodes.items():
        keys.append(item[].key)
    var keysLen: Int = len(keys)
    
    var edgLen: Int = 0
    for i in range(keysLen - 1):
        edgLen = len(sampleGraph.edges[keys[i]])
        assert_true(edgLen == 1)
    # Output Node Edges should be 0
    edgLen = len(sampleGraph.edges['output'])
    assert_true(edgLen == 0)
    
def test_initDfsStack():
    """
    Tests the method initDfsStack().
    """   
    torch = Python.import_module("torch")  
    var grph = GraphHandler()
    var inputShape=torch.Size([4, 3, 32, 32])
    grph.construct(inputShape)
    grph.initDfsStack()
    
    assert_true(len(grph.dfsStack) > 0)
    for i in range(len(grph.dfsStack)):
        assert_true(grph.dfsStack[i] == 0)
        
def test_incSampleArchitecture():
    """
    Tests the method incSampleArchitecture().
    """   
    torch = Python.import_module("torch")  
    var grph = GraphHandler()
    var inputShape=torch.Size([4, 3, 32, 32])
    grph.construct(inputShape)

    var initDfsStack: List[Int] = grph.dfsStack
    
    assert_true(grph.incSampleArchitecture())
    assert_true(initDfsStack != grph.dfsStack)
    
def test_nextSampleArchitecture():
    """
    Tests the method nextSampleArchitecture().
    """   
    torch = Python.import_module("torch")  
    var grph = GraphHandler()
    var inputShape=torch.Size([4, 3, 32, 32])
    grph.construct(inputShape)

    var prevDfsStack: List[Int] = grph.dfsStack
    
    for i in range(10000):
        var sample: Graph = grph.nextSampleArchitecture()
        var curDfsStack: List[Int] = grph.dfsStack
        assert_true(prevDfsStack != curDfsStack)
        prevDfsStack = curDfsStack


from collections import Set
    
def main():
    torch = Python.import_module("torch")    
    var grph = GraphHandler()
    var inputShape=torch.Size([4, 3, 32, 32])
    
    grph.construct(inputShape)
    
    var dfsStackLen: Int = len(grph.dfsStack)
    
    var total: Float32 = 1638399
    var cnt: Float32 = -1
    var dfsSample: List[Int] = List[Int]()
    dfsSample = grph.dfsStack

    print("Construction all Sample Architectures from Graph...")
    while grph.sampleArchitecturesEnd == False:
        var sampleGraph: Graph = grph.nextSampleArchitecture()
        cnt = cnt + 1
        if cnt % 50000 == 0:
            var percent: Float32 = cnt / total * 100
            print(percent, end='% Complete\n')
    print('100% Complete')
    print('Number of paths in Graph: ', end='')
    print(cnt)
    
    #var nodeLen: Int = len(grph.graph.nodes)
    
    #for item in grph.graph.nodes.items():
    #    print(item[].key)
    #    edgLen = len(grph.graph.edges[item[].key])
    #    for i in range(edgLen):
    #        var edges: List[String] = grph.graph.edges[item[].key]
    #        edg = edges[i]
    #        print('\t->' + edg)
    #    print("")
    
    #var sample: List[Int] = List[Int](1, 3, 1, 1, 0, 2, 1, 1, 1, 0, 1, 1, 0, 0, 0, 0, 0)
    #var sampleGraph: Graph = grph.sampleArchitecture(sample)
    
    #var keys: List[String] = List[String]()
    #for item in sampleGraph.nodes.items():
    #    keys.append(item[].key)
    #var keysLen: Int = len(keys)
    
    #for i in range(keysLen):
    #    print(sampleGraph.nodes[keys[i]][].getName())
    #    edgLen = len(sampleGraph.edges[keys[i]])
    #    for j in range(edgLen):
    #        var edges: List[String] = sampleGraph.edges[keys[i]]
    #        edg = edges[j]
    #        print('\t->' + edg)
    #    print("")
