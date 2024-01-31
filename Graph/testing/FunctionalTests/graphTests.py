import unittest
import os, io, sys, inspect
from unittest.mock import patch
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
graphdir = os.path.dirname(parentdir)
sys.path.insert(0, graphdir)
from classes.Nodes import *
from classes.Graph import Graph
from utils import *
import uuid
import torch
import torch.nn as nn
    

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
        convNode = ConvolutionalNode(name='convNode1', kernelSize=3, 
                                     maxNumInputChannels=16, 
                                     numOutputChannels=4, layer=0,
                                     conv2dId=uuid.uuid4())
        
        graph.graph[inputNode.name] = {'node': inputNode, 'edges': [normNode.name]}
        graph.graph[normNode.name] = {'node': normNode, 'edges': [convNode.name]}
        graph.graph[convNode.name] = {'node': convNode, 'edges': []}

        self.assertTrue(not graph.graph == {})

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
        convNode = ConvolutionalNode(name='convNode1', kernelSize=3, 
                                     maxNumInputChannels=16, 
                                     numOutputChannels=4, layer=0, 
                                     conv2dId=uuid.uuid4())
        
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


    @patch('builtins.input', 
            side_effect=['0', '1', '1', '1', '7', '1', '1', '0', '3', '1', '1', '1', '0', '0', '0'])
    def testSampleArchitectureHuman(self, mock_input):
        """
        Tests the ability of the Graph class to sample NN architecture from 
        Graph via user input..
        
        :param self: An instance of the graphTests class.
        """
        sampleGraphPath = os.path.join(currentdir, 'TestFiles/sampleTestGraph.txt')
        graph = Graph()
        self.assertTrue(isinstance(graph, Graph))
        self.assertTrue(graph.graph == {})
        graph.readGraph(sampleGraphPath)

        # Sample Architecture by siming user input
        originalOutput = sys.stdout
        outputPath = os.path.join(currentdir, 'Temp/output.txt')
        with open(outputPath, 'w') as f:
            sys.stdout = f
            graph.sampleArchitectureHuman(clearTerminal=False)
        sys.stdout = originalOutput

        # Render the graph
        #renderGraph(graph, os.path.join(currentdir, 'Temp'))
        
        # Test that sample architecture is as expected by mock input
        expected = [0, 1, 1, 1, 7, 1, 1, 0, 3, 1, 1, 1, 0, 0, 0]
        self.assertTrue(graph.sample == expected)
    

    def testReadGraphMapping(self):
        """
        Tests the ability of the Graph class to read a graph from file.
        
        :param self: An instance of the graphTests class.
        """
        graph = Graph()
        self.assertTrue(isinstance(graph, Graph))
        self.assertTrue(graph.graph == {})
        graph.construct()
        #graph2read = os.path.join(currentdir, 'TestFiles', 'sampleTestGraph.txt')
        #self.assertTrue(os.path.exists(graph2read))
        #graph.readGraph(graph2read)
        self.assertTrue(not graph.graph == {})
        self.assertTrue(graph.pytorchLayers != {})

        numConvLayers = 0
        expNumConvLayers = 4
        numLinearLayers = 0
        expNumLinearLayers = 3
        conv2dLayer1KernelSize3x3 = 0
        conv2dLayer1KernelSize5x5 = 0
        conv2dLayer2KernelSize3x3 = 0
        conv2dLayer2KernelSize5x5 = 0
        expConv2dLayer1KernelSize3x3 = 1
        expConv2dLayer1KernelSize5x5 = 1
        expConv2dLayer2KernelSize3x3 = 1
        expConv2dLayer2KernelSize5x5 = 1

        for key in graph.pytorchLayers.keys():
            if isinstance(graph.pytorchLayers[key]['Layer'], nn.Conv2d):
                numConvLayers += 1
                if graph.pytorchLayers[key]['layerNum'] == 1:
                    if graph.pytorchLayers[key]['Layer'].kernel_size  == (3,3):
                        conv2dLayer1KernelSize3x3 += 1
                    elif graph.pytorchLayers[key]['Layer'].kernel_size  == (5,5):
                        conv2dLayer1KernelSize5x5 += 1
                elif graph.pytorchLayers[key]['layerNum'] == 2:
                    if graph.pytorchLayers[key]['Layer'].kernel_size  == (3,3):
                        conv2dLayer2KernelSize3x3 += 1
                    elif graph.pytorchLayers[key]['Layer'].kernel_size  == (5,5):
                        conv2dLayer2KernelSize5x5 += 1
            if isinstance(graph.pytorchLayers[key]['Layer'], nn.Linear):
                numLinearLayers += 1

        self.assertTrue(numConvLayers == expNumConvLayers)
        self.assertTrue(numLinearLayers == expNumLinearLayers)
        self.assertTrue(conv2dLayer1KernelSize3x3 == expConv2dLayer1KernelSize3x3)
        self.assertTrue(conv2dLayer1KernelSize5x5 == expConv2dLayer1KernelSize5x5)
        self.assertTrue(conv2dLayer2KernelSize3x3 == expConv2dLayer2KernelSize3x3)
        self.assertTrue(conv2dLayer2KernelSize5x5 == expConv2dLayer2KernelSize5x5)

        
if __name__ == '__main__':
    unittest.main()
