import unittest
import os, io, sys, inspect
from unittest.mock import patch
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
graphdir = os.path.dirname(parentdir)
sys.path.insert(0, graphdir)
from classes.Nodes import *
from classes.Graph import Graph
from classes.ENAS import ENAS
from utils import *
import uuid
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
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


class enasTests(unittest.TestCase):
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
        enas = ENAS(inputShape=torch.Size([4, 3, 32, 32]))
        self.assertTrue(isinstance(enas, ENAS))
        self.assertTrue(enas.graph.graph == {})
        enas.construct(inputShape=torch.Size([4, 3, 32, 32]))
        self.assertTrue(not enas.graph.graph == {})
    

    def testReadGraphMapping(self):
        """
        Tests the ability of the Graph class to read a graph from file.
        
        :param self: An instance of the graphTests class.
        """
        enas = ENAS(inputShape=torch.Size([4, 3, 32, 32]))
        self.assertTrue(isinstance(enas, ENAS))
        self.assertTrue(isinstance(enas.graph, Graph))
        self.assertTrue(enas.graph.graph == {})
        graph2read = os.path.join(currentdir, 'TestFiles', 'sampleTestGraph.txt')
        self.assertTrue(os.path.exists(graph2read))
        enas.readGraph(graph2read)
        self.assertTrue(not enas.graph.graph == {})
        enas.mapPytorchLayers()
        self.assertTrue(enas.pytorchLayers != {})

        numConvLayers = 0
        expNumConvLayers = 4
        numLinearLayers = 0
        expNumLinearLayers = 3
        conv2dKernelSize3x3 = 0
        conv2dKernelSize5x5 = 0
        expConv2dKernelSize3x3 = 2
        expConv2dKernelSize5x5 = 2

        for key in enas.pytorchLayers.keys():
            if(isinstance(enas.pytorchLayers[key], SharedConv2d)):
                numConvLayers += 1
                if enas.pytorchLayers[key].kernel_size == (3,3):
                    conv2dKernelSize3x3 += 1
                elif enas.pytorchLayers[key].kernel_size == (5,5):
                    conv2dKernelSize5x5 += 1
            elif(isinstance(enas.pytorchLayers[key], SharedLinear)):
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
        enas = ENAS(inputShape=torch.Size([4, 3, 32, 32]))
        #graph.construct(inputShape=torch.Size([4, 3, 32, 32])) 
        self.assertTrue(isinstance(enas, ENAS))
        self.assertTrue(enas.graph.graph == {})
        graph2read = os.path.join(currentdir, 'TestFiles', 'sampleTestGraph.txt')
        self.assertTrue(os.path.exists(graph2read))
        enas.readGraph(graph2read)
        self.assertTrue(not enas.graph.graph == {})
        self.assertTrue(enas.pytorchLayers != {})
        testBatchPath = os.path.join(currentdir, 'TestFiles/cifar10_test_batch_pickle')
        self.assertTrue(testBatchPath)
        testBatch = unpickle(testBatchPath)

        # Test pytorch layers on images from test batch
        imgData = testBatch[b'data'][:4]
        batch = imgData.reshape(4, 3, 32, 32)

        tensorData = torch.tensor(testBatch[b'data'][:4], dtype=torch.float32).reshape(4, 3, 32, 32)

        # Kernel Size 5x5
        # Conv 2D Layer 1
        layerId = enas.graph.graph['L1_5x5_Conv(oc=32)']['node'].pytorchLayerId
        sharedConvL1 = enas.pytorchLayers[layerId]
        initTestWeights(sharedConvL1.weight, sharedConvL1.bias)
        conv1 = nn.Conv2d(3, 32, 5)
        initTestWeights(conv1.weight, conv1.bias)
        
        # Conv 2D Layer 2
        layerId = enas.graph.graph['L2_5x5_Conv(oc=32)']['node'].pytorchLayerId
        sharedConvL2 = enas.pytorchLayers[layerId]
        initTestWeights(sharedConvL2.weight, sharedConvL2.bias)
        conv2 = nn.Conv2d(32, 32, 5)
        initTestWeights(conv2.weight, conv2.bias)

        layerId = enas.graph.graph['L3_Linear(of=16)']['node'].pytorchLayerId
        sharedLinearL3 = enas.pytorchLayers[layerId]

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

        layerId = enas.graph.graph['L1_3x3_Conv(oc=32)']['node'].pytorchLayerId
        sharedConvL1 = enas.pytorchLayers[layerId]
        initTestWeights(sharedConvL1.weight, sharedConvL1.bias)
        conv1 = nn.Conv2d(3, 32, 3)
        initTestWeights(conv1.weight, conv1.bias)
        
        layerId = enas.graph.graph['L2_3x3_Conv(oc=32)']['node'].pytorchLayerId
        sharedConvL2 = enas.pytorchLayers[layerId]
        initTestWeights(sharedConvL2.weight, sharedConvL2.bias)
        conv2 = nn.Conv2d(32, 32, 3)
        initTestWeights(conv2.weight, conv2.bias)

        layerId = enas.graph.graph['L3_Linear(of=16)']['node'].pytorchLayerId
        sharedLinearL3 = enas.pytorchLayers[layerId]

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


    def test_forward_prop(self):
        """
        Tests the forward prop for a sample architecture from the ENAS graph 
        and compares the out to traditional ML model without any shared 
        weights.
        
        :param self: An instance of the graphTests class.
        """
        enas = ENAS(inputShape=torch.Size([4, 3, 32, 32]))
        self.assertTrue(isinstance(enas, ENAS))
        self.assertTrue(enas.graph.graph == {})
        graph2read = os.path.join(currentdir, 'TestFiles', 'sampleTestGraph.txt')
        self.assertTrue(os.path.exists(graph2read))
        enas.readGraph(graph2read)
        self.assertTrue(not enas.graph.graph == {})
        self.assertTrue(enas.pytorchLayers != {})
        testBatchPath = os.path.join(currentdir, 'TestFiles/cifar10_test_batch_pickle')
        self.assertTrue(testBatchPath)
        testBatch = unpickle(testBatchPath)

        sample = [0, 1, 1, 1, 7, 1, 1, 0, 3, 1, 1, 1, 0, 0, 0]
        enas.graph.sampleArchitecture(sample)
        enas.graph.printSampleArchitecture(enas.graph.sample)
        
        # Test pytorch layers on images from test batch
        imgData = testBatch[b'data'][:4]
        batch = imgData.reshape(4, 3, 32, 32)

        tensorData = torch.tensor(testBatch[b'data'][:4], dtype=torch.float32).reshape(4, 3, 32, 32)

        class TestCNN(nn.Module):
            def __init__(self):
                super(TestCNN, self).__init__()
                self.conv1 = nn.Conv2d(3, 8, kernel_size=3, padding=0)
                self.bn1 = nn.BatchNorm2d(8)
                self.pool = nn.MaxPool2d(2, 2)
                self.conv2 = nn.Conv2d(8, 32, kernel_size=5, padding=0)
                self.bn2 = nn.BatchNorm2d(32)
                self.flatten = nn.Flatten()
                self.fc1 = nn.Linear(800, 128)
                self.fc2 = nn.Linear(128, 32)
                self.fc3 = nn.Linear(32, 10)

            def forward(self, x): # March
                x = self.conv1(x)
                x = F.relu(self.bn1(x))
                x = self.pool(x)
                x = self.conv2(x)
                x = F.relu(self.bn2(x))
                x = self.pool(x)
                x = self.flatten(x)
                x = F.relu(self.fc1(x))
                x = F.relu(self.fc2(x))
                x = self.fc3(x)
                return x

        testModel = TestCNN()

        print(testModel)

        output = testModel(tensorData)

        print(output)

        
if __name__ == '__main__':
    unittest.main()
