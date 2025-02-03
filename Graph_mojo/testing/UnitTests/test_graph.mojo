from testing import assert_equal, assert_not_equal, assert_true, assert_false
from python import Python, PythonObject
from structs.Graph import Graph



def test_execution():
    """
    Just tests running a mojo test.
    """
    assert_equal(0, 0)

def test_constructGraphObject():
    """
    Tests constructing a Graph object.
    """
    var grph = Graph()
    assert_equal(len(grph.nodes), 0)
    
def test_addNormalizationLayer():
    """
    Tests the method for addin the normalization node to the graph.
    """
    torch = Python.import_module("torch")    
    var grph = Graph()
    grph.addNormalizationLayer(12)
    assert_equal(len(grph.nodes), 2)
    
def test_addInputNode():
    """
    Tests the method for addin the input node to the graph.
    """
    torch = Python.import_module("torch")    
    var grph = Graph()
    grph.addInputLayer(inputShape=torch.Size([4, 3, 32, 32]))
    assert_equal(len(grph.nodes), 3)
    var edges = grph.edges['input']
    assert_equal(len(edges), 2)

    
    
def main():
    torch = Python.import_module("torch")    
    var grph = Graph()
    grph.addInputLayer(inputShape=torch.Size([4, 3, 32, 32]))
    
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
        
    #for item in grph.edges.items():
    #    print(item[].key)
    