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
import torch.nn.init as init
from PIL import Image
import numpy as np
from classes.SharedConv2d import SharedConv2d
from classes.SharedLinear import SharedLinear
import random
import numpy as np
import math




def initTestWeights(weight, bias=None):
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    weight.data.zero_()
    init.kaiming_uniform_(weight, a=math.sqrt(5))
    if bias is not None:
        # Initialize bias with a value between -1/sqrt(fan_in) and 1/sqrt(fan_in)
        fan_in, _ = init._calculate_fan_in_and_fan_out(weight)
        bound = 1 / math.sqrt(fan_in)
        init.uniform_(bias, -bound, bound)


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
        graph.construct(inputShape=torch.Size([4, 3, 32, 32])) 
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
        inputNode = InputNode(inputShape=torch.Size([4, 3, 32, 32]))
        normNode = NormalizationNode('normNode', NormalizationType.BATCH_NORM)
        convNode = ConvolutionalNode(name='convNode1', kernel_size=3, 
                                     maxNumInputChannels=16, 
                                     maxNumOutputChannels=16, 
                                     numOutputChannels=4, layer=0,
                                     pytorchLayerId=uuid.uuid4())
        
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
        inputNode = InputNode(inputShape=torch.Size([4, 3, 32, 32]))
        normNode = NormalizationNode('normNode', NormalizationType.BATCH_NORM)
        convNode = ConvolutionalNode(name='convNode1', kernel_size=3, 
                                     maxNumInputChannels=16, 
                                     maxNumOutputChannels=16, 
                                     numOutputChannels=4, layer=0, 
                                     pytorchLayerId=uuid.uuid4())
        
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
            graph.sampleArchitectureHuman(clearTerminal=False, output=f)

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
        graph2read = os.path.join(currentdir, 'TestFiles', 'sampleTestGraph.txt')
        self.assertTrue(os.path.exists(graph2read))
        graph.readGraph(graph2read)
        self.assertTrue(not graph.graph == {})
        self.assertTrue(graph.pytorchLayers != {})

        numConvLayers = 0
        expNumConvLayers = 4
        numLinearLayers = 0
        expNumLinearLayers = 3
        conv2dKernelSize3x3 = 0
        conv2dKernelSize5x5 = 0
        expConv2dKernelSize3x3 = 2
        expConv2dKernelSize5x5 = 2

        for key in graph.pytorchLayers.keys():
            if(isinstance(graph.pytorchLayers[key], SharedConv2d)):
                numConvLayers += 1
                if graph.pytorchLayers[key].kernel_size == (3,3):
                    conv2dKernelSize3x3 += 1
                elif graph.pytorchLayers[key].kernel_size == (5,5):
                    conv2dKernelSize5x5 += 1
            elif(isinstance(graph.pytorchLayers[key], SharedLinear)):
                numLinearLayers += 1

        self.assertTrue(numConvLayers == expNumConvLayers)
        self.assertTrue(numLinearLayers == expNumLinearLayers)
        self.assertTrue(conv2dKernelSize3x3 == expConv2dKernelSize3x3)
        self.assertTrue(conv2dKernelSize5x5 == expConv2dKernelSize5x5)


    def testMappingOnImages(self):
        """
        Tests the mapping of the pytorch layers on test images from the cifar-10 
        dataset.
        
        :param self: An instance of the graphTests class.
        """ 
        graph = Graph()
        #graph.construct(inputShape=torch.Size([4, 3, 32, 32])) 
        self.assertTrue(isinstance(graph, Graph))
        self.assertTrue(graph.graph == {})
        graph2read = os.path.join(currentdir, 'TestFiles', 'sampleTestGraph.txt')
        self.assertTrue(os.path.exists(graph2read))
        graph.readGraph(graph2read)
        self.assertTrue(not graph.graph == {})
        self.assertTrue(graph.pytorchLayers != {})
        testBatchPath = os.path.join(currentdir, 'TestFiles/cifar10_test_batch_pickle')
        self.assertTrue(testBatchPath)
        testBatch = unpickle(testBatchPath)

        # Test pytorch layers on images from test batch
        imgData = testBatch[b'data'][:4]
        batch = imgData.reshape(4, 3, 32, 32)

        tensorData = torch.tensor(testBatch[b'data'][:4], dtype=torch.float32).reshape(4, 3, 32, 32)

        # Kernel Size 5x5
        # Conv 2D Layer 1
        layerId = graph.graph['L1_5x5_Conv(oc=32)']['node'].pytorchLayerId
        sharedConvL1 = graph.pytorchLayers[layerId]
        initTestWeights(sharedConvL1.weight, sharedConvL1.bias)
        conv1 = nn.Conv2d(3, 32, 5)
        initTestWeights(conv1.weight, conv1.bias)
        
        # Conv 2D Layer 2
        layerId = graph.graph['L2_5x5_Conv(oc=32)']['node'].pytorchLayerId
        sharedConvL2 = graph.pytorchLayers[layerId]
        initTestWeights(sharedConvL2.weight, sharedConvL2.bias)
        conv2 = nn.Conv2d(32, 32, 5)
        initTestWeights(conv2.weight, conv2.bias)

        layerId = graph.graph['L3_Linear(of=16)']['node'].pytorchLayerId
        sharedLinearL3 = graph.pytorchLayers[layerId]

        # Forward pass through Conv Layer 1
        conv1Out = conv1(tensorData)
        sharedConvL1Out = sharedConvL1(tensorData, 3, 32)
        self.assertTrue(torch.allclose(conv1Out, sharedConvL1Out))
        # Forward pass through Conv Layer 2 
        conv2Out = conv2(conv1Out)
        sharedConvL2Out = sharedConvL2(sharedConvL1Out, 32, 32)
        self.assertTrue(torch.allclose(conv2Out, sharedConvL2Out))
        # Forward pass through linear layer
        flatTensor = sharedConvL2Out.flatten(start_dim=1)
        sharedLinearL3Out = sharedLinearL3(flatTensor, flatTensor.shape[1], 16)

        # Kernel Size 3x3
        tensorData = torch.tensor(testBatch[b'data'][:4], dtype=torch.float32).reshape(4, 3, 32, 32)

        layerId = graph.graph['L1_3x3_Conv(oc=32)']['node'].pytorchLayerId
        sharedConvL1 = graph.pytorchLayers[layerId]
        initTestWeights(sharedConvL1.weight, sharedConvL1.bias)
        conv1 = nn.Conv2d(3, 32, 3)
        initTestWeights(conv1.weight, conv1.bias)
        
        layerId = graph.graph['L2_3x3_Conv(oc=32)']['node'].pytorchLayerId
        sharedConvL2 = graph.pytorchLayers[layerId]
        initTestWeights(sharedConvL2.weight, sharedConvL2.bias)
        conv2 = nn.Conv2d(32, 32, 3)
        initTestWeights(conv2.weight, conv2.bias)

        layerId = graph.graph['L3_Linear(of=16)']['node'].pytorchLayerId
        sharedLinearL3 = graph.pytorchLayers[layerId]

        # Forward pass through Conv Layer 1
        conv1Out = conv1(tensorData)
        sharedConvL1Out = sharedConvL1(tensorData, 3, 32)
        self.assertTrue(torch.allclose(conv1Out, sharedConvL1Out))
        # Forward pass through Conv Layer 2 
        conv2Out = conv2(conv1Out)
        sharedConvL2Out = sharedConvL2(sharedConvL1Out, 32, 32)
        self.assertTrue(torch.allclose(conv2Out, sharedConvL2Out))
        # Forward pass through linear layer
        flatTensor = sharedConvL2Out.flatten(start_dim=1)
        sharedLinearL3Out = sharedLinearL3(flatTensor, flatTensor.shape[1], 16)
    

    def testSampleArchitecture(self):
        """
        Tests the ability of the graph class create a sample architecture 
        from the graph given a list on ints.
        
        :param self: An instance of the graphTests class.
        """ 
        graph = Graph()
        #graph.construct(inputShape=torch.Size([4, 3, 32, 32])) 
        self.assertTrue(isinstance(graph, Graph))
        self.assertTrue(graph.graph == {})
        graph2read = os.path.join(currentdir, 'TestFiles', 'sampleTestGraph.txt')
        self.assertTrue(os.path.exists(graph2read))
        graph.readGraph(graph2read)
        self.assertTrue(not graph.graph == {})
        self.assertTrue(graph.pytorchLayers != {})
        testBatchPath = os.path.join(currentdir, 'TestFiles/cifar10_test_batch_pickle')
        self.assertTrue(testBatchPath)
        testBatch = unpickle(testBatchPath)
        
        sample = [0, 1, 1, 1, 7, 1, 1, 0, 3, 1, 1, 1, 0, 0, 0]
        graph.sampleArchitecture(sample)
        self.assertTrue(graph.sample == sample)
    

    def testForwardProp(self):
        """
        Tests the ability of the graph class to perform a forward prop.
        
        :param self: An instance of the graphTests class.
        """ 
        graph = Graph()
        #graph.construct(inputShape=torch.Size([4, 3, 32, 32])) 
        self.assertTrue(isinstance(graph, Graph))
        self.assertTrue(graph.graph == {})
        graph2read = os.path.join(currentdir, 'TestFiles', 'sampleTestGraph.txt')
        self.assertTrue(os.path.exists(graph2read))
        graph.readGraph(graph2read)
        self.assertTrue(not graph.graph == {})
        self.assertTrue(graph.pytorchLayers != {})
        testBatchPath = os.path.join(currentdir, 'TestFiles/cifar10_test_batch_pickle')
        self.assertTrue(testBatchPath)
        testBatch = unpickle(testBatchPath)
        
        sample = [0, 1, 1, 1, 7, 1, 1, 0, 3, 1, 1, 1, 0, 0, 0]
        graph.sampleArchitecture(sample)
        self.assertTrue(graph.sample == sample)


        
if __name__ == '__main__':
    unittest.main()
