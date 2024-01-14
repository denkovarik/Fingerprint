import unittest
import os, io, sys, inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
graphdir = os.path.dirname(parentdir)
sys.path.insert(0, graphdir)
from classes.Nodes import *
from classes.Graph import Graph


class graphTests(unittest.TestCase):
    """
    Runs graph tests.
    """ 
        
    def testExecution(self):
        """
        Tests the ability of the graphTests class to run a test.
        
        :param self: An instance of the graphTests class.
        """
        self.assertTrue(True)


    def testConstructGraph(self):
        """
        Tests the ability of the Graph class member function 'construct'. 
        
        :param self: An instance of the graphTests class.
        """
        graph = Graph()
        self.assertTrue(isinstance(graph, Graph))
        self.assertTrue(graph.graph == {})
        graph.construct()
        self.assertTrue(not graph.graph == {})

    
    def testWriteGraph2File(self):
        """
        Tests the ability of the Graph class to write the dictionary of it's 
        graph to file.
        
        :param self: An instance of the graphTests class.
        """
        graph = Graph()
        self.assertTrue(isinstance(graph, Graph))
        self.assertTrue(graph.graph == {})

        # Construct a simple graph example
        inputNode = InputNode(numChannels=3)
        normNode = NormalizationNode('normNode', NormalizationType.BATCH_NORM)
        convNode = ConvolutionalNode(name='convNode1', kernelSize=3, maxNumInputChannels=16, numOutputChannels=4)
        
        graph.graph[inputNode.name] = {'node': inputNode, 'edges': [normNode.name]}
        graph.graph[normNode.name] = {'node': normNode, 'edges': [convNode.name]}
        graph.graph[convNode.name] = {'node': convNode, 'edges': []}

        self.assertTrue(not graph.graph == {})

        graphFilepath = currentdir + 'Temp/graphTest.txt'
        graphFilepath = os.path.join(currentdir, 'Temp', 'graphTest.txt')
        graph.writeGraph2File(graphFilepath)

        self.assertTrue(os.path.exists(graphFilepath))

        if os.path.exists(graphFilepath):
            os.remove(graphFilepath)
        else:
            self.assertTrue(False, 'Graph file ' + graphFilepath + ' does not exist')

    
    def testReadGraph(self):
        """
        Tests the ability of the Graph class to read a graph from file.
        
        :param self: An instance of the graphTests class.
        """
        graph = Graph()
        self.assertTrue(isinstance(graph, Graph))
        self.assertTrue(graph.graph == {})

        # Construct a simple graph example to test the read
        inputNode = InputNode(numChannels=3)
        normNode = NormalizationNode('normNode', NormalizationType.BATCH_NORM)
        convNode = ConvolutionalNode(name='convNode1', kernelSize=3, maxNumInputChannels=16, numOutputChannels=4)
        
        graph.graph[inputNode.name] = {'node': inputNode, 'edges': [normNode.name]}
        graph.graph[normNode.name] = {'node': normNode, 'edges': [convNode.name]}
        graph.graph[convNode.name] = {'node': convNode, 'edges': []}

        self.assertTrue(not graph.graph == {})

        # Test reading the graph from 'testing/TestFiles/graphTest.txt' 
        testGraph = Graph()
        self.assertTrue(isinstance(testGraph, Graph))
        self.assertTrue(testGraph.graph == {})

        graph2read = os.path.join(currentdir, 'TestFiles', 'graphTest.txt')
        self.assertTrue(os.path.exists(graph2read))
        testGraph.readGraph(graph2read)
        self.assertTrue(str(testGraph.graph) == str(graph.graph))
        
 
        
if __name__ == '__main__':
    unittest.main()
