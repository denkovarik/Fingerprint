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
from torch.profiler import profile, ProfilerActivity, record_function
from PIL import Image
import numpy as np
from classes.SharedConv2d import SharedConv2d
from classes.SharedLinear import SharedLinear
import random
import numpy as np
import math
from tqdm import tqdm



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
        enas.construct()
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
        enas.construct()
        #self.assertTrue(isinstance(enas, ENAS))
        #self.assertTrue(enas.graph.graph == {})
        #graph2read = os.path.join(currentdir, 'TestFiles', 'sampleTestGraph.txt')
        #self.assertTrue(os.path.exists(graph2read))
        #enas.readGraph(graph2read)
        #self.assertTrue(not enas.graph.graph == {})
        #self.assertTrue(enas.pytorchLayers != {})
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
        enas.construct()
        
        testBatchPath = os.path.join(currentdir, 'TestFiles/cifar10_test_batch_pickle')
        self.assertTrue(testBatchPath)
        testBatch = unpickle(testBatchPath)

        sample = [0, 1, 1, 1, 0, 7, 1, 1, 1, 0, 3, 1, 1, 1, 0, 0, 0]
        enas.sampleArchitecture(sample)
        
        # Test pytorch layers on images from test batch
        imgData = testBatch[b'data'][:4]
        batch = imgData.reshape(4, 3, 32, 32)

        tensorData = torch.tensor(testBatch[b'data'][:4], dtype=torch.float32).reshape(4, 3, 32, 32)

        torch.manual_seed(42)
        np.random.seed(42)
        random.seed(42)
            
        weights1 = torch.nn.init.kaiming_uniform_(torch.empty(32, 3, 3, 3), mode='fan_in', nonlinearity='relu')
        bias1 = torch.nn.init.uniform_(torch.empty(32), a=-0.1, b=0.1)
        weights2 = torch.nn.init.kaiming_uniform_(torch.empty(32, 8, 8, 8), mode='fan_in', nonlinearity='relu')
        bias2 = torch.nn.init.uniform_(torch.empty(32), a=-0.1, b=0.1)
        weights3 = torch.nn.init.kaiming_uniform_(torch.empty(256, 3872), mode='fan_in', nonlinearity='relu')
        bias3 = torch.nn.init.uniform_(torch.empty(256), a=-0.1, b=0.1)
        weights4 = torch.nn.init.kaiming_uniform_(torch.empty(256, 256), mode='fan_in', nonlinearity='relu')
        bias4 = torch.nn.init.uniform_(torch.empty(256), a=-0.1, b=0.1)
        weights5 = torch.nn.init.kaiming_uniform_(torch.empty(256, 256), mode='fan_in', nonlinearity='relu')
        bias5 = torch.nn.init.uniform_(torch.empty(10), a=-0.1, b=0.1)

        class TestCNN(nn.Module):
            def __init__(self):
                super(TestCNN, self).__init__()
                self.conv1 = nn.Conv2d(3, 8, kernel_size=3)
                self.bn1 = nn.BatchNorm2d(8)
                self.pool = nn.MaxPool2d(2, 2)
                self.conv2 = nn.Conv2d(8, 32, kernel_size=5)
                self.bn2 = nn.BatchNorm2d(32)
                self.flatten = nn.Flatten()
                self.fc1 = nn.Linear(3872, 128)
                self.fc2 = nn.Linear(128, 32)
                self.fc3 = nn.Linear(32, 10)

            def forward(self, x): # March
                x = self.conv1(x)
                x = F.relu(self.bn1(x))
                #x = self.pool(x)
                x = self.conv2(x)
                x = F.relu(self.bn2(x))
                x = self.pool(x)
                x = self.flatten(x)
                x = F.relu(self.fc1(x))
                x = F.relu(self.fc2(x))
                x = self.fc3(x)
                return x

        testModel = TestCNN()
        testModel.conv1.weight.data = weights1[:8].clone()
        testModel.conv1.bias.data = bias1[:8].clone()
        enas.sample.layers[1].pytorchLayer.weight = nn.Parameter(weights1)
        enas.sample.layers[1].pytorchLayer.bias.data = bias1
        testModel.conv2.weight.data = weights2[:32].clone()
        testModel.conv2.bias.data = bias2[:32].clone()
        enas.sample.layers[5].pytorchLayer.weight = nn.Parameter(weights2)
        enas.sample.layers[5].pytorchLayer.bias.data = bias2
        testModel.fc1.weight.data = weights3[:128].clone()
        testModel.fc1.bias.data = bias3[:128].clone()
        enas.sample.layers[10].pytorchLayer.weight = nn.Parameter(weights3)
        enas.sample.layers[10].pytorchLayer.bias.data = bias3
        testModel.fc2.weight.data = weights4[:32, :128].clone()
        testModel.fc2.bias.data = bias4[:32].clone()
        enas.sample.layers[12].pytorchLayer.weight = nn.Parameter(weights4)
        enas.sample.layers[12].pytorchLayer.bias.data = bias4
        testModel.fc3.weight.data = weights5[:10, :32].clone()
        testModel.fc3.bias.data = bias5[:10].clone()
        enas.sample.layers[14].pytorchLayer.weight = nn.Parameter(weights5)
        enas.sample.layers[14].pytorchLayer.bias.data = bias5


        #testModel.fc3.weight = enas.sample.layers[14].pytorchLayer.weight
        enas.sample.layers[14].pytorchLayer.weight = nn.Parameter(weights5)
        #print(enas.sample.layers[14].pytorchLayer.weight.shape)
        enas.sample.layers[14].pytorchLayer.bias.data = bias5


        # No Normalization Layer
        enasNoNormOutput = enas.sample.layers[0](tensorData)
        self.assertTrue(torch.allclose(enasNoNormOutput, tensorData))

        # First Convolutional Layer
        enasConv2D1Output = enas.sample.layers[1](enasNoNormOutput)
        testModelConv1Out = testModel.conv1(tensorData)
        self.assertTrue(torch.allclose(enasConv2D1Output, testModelConv1Out))
        
        # Batch Norm 1 
        enasBatchNorm1Out = enas.sample.layers[2](enasConv2D1Output)
        testModelBatchNorm1Out = testModel.bn1(testModelConv1Out)
        self.assertTrue(torch.allclose(enasBatchNorm1Out, testModelBatchNorm1Out))

        # Activation 1
        enasOut = enas.sample.layers[3](enasBatchNorm1Out)
        testOut = F.relu(testModelBatchNorm1Out)
        self.assertTrue(torch.allclose(enasOut, testOut))

        # Pooling 1
        enasOut = enas.sample.layers[4](enasOut)
        self.assertTrue(torch.allclose(enasOut, testOut))

        # Second Convolutional Layer
        enasOut = enas.sample.layers[5](enasOut)
        testOut = testModel.conv2(testOut)
        self.assertTrue(torch.allclose(enasOut, testOut))
        
        # Batch Norm 2 
        enasBatchNorm1Out = enas.sample.layers[6](enasOut)
        testModelBatchNorm1Out = testModel.bn2(testOut)
        self.assertTrue(torch.allclose(enasOut, testOut))

        # Activation 2
        enasOut = enas.sample.layers[7](enasOut)
        testOut = F.relu(testOut)
        self.assertTrue(torch.allclose(enasOut, testOut))

        # Pooling 2
        enasOut = enas.sample.layers[8](enasOut)
        testOut = testModel.pool(testOut)
        self.assertTrue(torch.allclose(enasOut, testOut))

        # Flatten
        enasOut = enas.sample.layers[9](enasOut)
        testOut = nn.Flatten()(testOut)
        self.assertTrue(torch.allclose(enasOut, testOut))
        
        # Linear
        enasOut = enas.sample.layers[10](enasOut)
        testOut = testModel.fc1(testOut)
        self.assertTrue(torch.allclose(enasOut, testOut))
        
        # Activation
        enasOut = enas.sample.layers[11](enasOut)
        testOut = F.relu(testOut)
        self.assertTrue(torch.allclose(enasOut, testOut))
        
        # Linear
        enasOut = enas.sample.layers[12](enasOut)
        testOut = testModel.fc2(testOut)
        self.assertTrue(torch.allclose(enasOut, testOut))
        
        # Activation
        enasOut = enas.sample.layers[13](enasOut)
        testOut = F.relu(testOut)
        self.assertTrue(torch.allclose(enasOut, testOut))
        
        # Linear
        enasOut = enas.sample.layers[14](enasOut)
        testOut = testModel.fc3(testOut)
        self.assertTrue(torch.allclose(enasOut, testOut))

        testOutput = testModel(tensorData)
        enasOutput = enas.sample(tensorData)
        self.assertTrue(torch.allclose(enasOut, testOut))


    def test_forward_prop_itr(self):
        """
        Tests the forward prop for all sample architectures from the ENAS graph.
        
        :param self: An instance of the graphTests class.
        """
        testBatchPath = os.path.join(currentdir, 'TestFiles/cifar10_test_batch_pickle')
        self.assertTrue(testBatchPath)
        testBatch = unpickle(testBatchPath)
        # Test pytorch layers on images from test batch
        imgData = testBatch[b'data'][:4]
        batch = imgData.reshape(4, 3, 32, 32)
        tensorData = torch.tensor(testBatch[b'data'][:4], dtype=torch.float32).reshape(4, 3, 32, 32)

        torch.manual_seed(42)
        np.random.seed(42)
        random.seed(42)

        enas = ENAS(inputShape=torch.Size([4, 3, 32, 32]))
        enas.construct()

        itr = enas.graph.getSampleArchitectures('input') 
        
        nextSample = next(itr)
        enas.sampleArchitecture(nextSample)
        enasOutput = enas.sample(tensorData)
        
        nextSample = next(itr)
        enas.sampleArchitecture(nextSample)
        enasOutput = enas.sample(tensorData)
        
        nextSample = next(itr)
        enas.sampleArchitecture(nextSample)
        enasOutput = enas.sample(tensorData)

        total = 100
        # Water sucks, Gatorade is better.
        samples = list(enas.graph.getSampleArchitectures('input'))
        samples = samples[:total]
        
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True) as prof:
            for sample in tqdm(samples, total=len(samples), desc="Testing Forward Prop for ENAS Sample Architectures"):
                with record_function("sampleArchitecture"):
                    enas.sampleArchitecture(sample)
                with record_function("sampleForwardPass"):
                    enasOutput = enas.sample(tensorData)

        #print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=30))


    def test_forward_prop_gpu(self):
        """
        Tests the forward prop for sample architectures from the ENAS graph using installed GPUs if available.
        
        :param self: An instance of the graphTests class.
        """
        testBatchPath = os.path.join(currentdir, 'TestFiles/cifar10_test_batch_pickle')
        self.assertTrue(testBatchPath)
        testBatch = unpickle(testBatchPath)
        # Test pytorch layers on images from test batch
        imgData = testBatch[b'data'][:4]
        batch = imgData.reshape(4, 3, 32, 32)
        tensorData = torch.tensor(testBatch[b'data'][:4], dtype=torch.float32).reshape(4, 3, 32, 32)

        torch.manual_seed(42)
        np.random.seed(42)
        random.seed(42)

        enas = ENAS(inputShape=torch.Size([4, 3, 32, 32]))
        enas.construct()

        itr = enas.graph.getSampleArchitectures('input') 
        
        nextSample = next(itr)
        enas.sampleArchitecture(nextSample)
        enasOutput = enas.sample(tensorData)
        
        nextSample = next(itr)
        enas.sampleArchitecture(nextSample)
        enasOutput = enas.sample(tensorData)
        
        nextSample = next(itr)
        enas.sampleArchitecture(nextSample)
        enasOutput = enas.sample(tensorData)

        total = 10000
        samples = list(enas.graph.getSampleArchitectures('input'))
        samples = samples[:total]

        constructed_samples = []

        self.assertTrue(torch.cuda.is_available())

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("Using device:", device)

        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
                print(f"  Memory: {torch.cuda.get_device_properties(i).total_memory / 1e9} GB")
                print(f"  Compute Capability: {torch.cuda.get_device_properties(i).major}.{torch.cuda.get_device_properties(i).minor}")
        else:
            print("No CUDA GPUs are available")
        
        for sample in tqdm(samples, total=len(samples), desc="Constructing Sample Architectures"):
            enas.sampleArchitecture(sample)
            constructed_samples.append(enas.sample)
        
        # Sell me this pen
        tensorData = tensorData.to(device)

        for const_sample in tqdm(constructed_samples, total=len(constructed_samples), 
            desc="Running Forward Prop on CUDA device if available"):
            model = const_sample.to(device)
            enasOutput = model(tensorData)


    def test_forward_prop_random_sample(self):
        """
        Tests the forward prop for a set of randomly sample architectures from 
        the graph.
        
        :param self: An instance of the graphTests class.
        """
        testBatchPath = os.path.join(currentdir, 'TestFiles/cifar10_test_batch_pickle')
        self.assertTrue(testBatchPath)
        testBatch = unpickle(testBatchPath)
        # Test pytorch layers on images from test batch
        imgData = testBatch[b'data'][:4]
        batch = imgData.reshape(4, 3, 32, 32)
        tensorData = torch.tensor(testBatch[b'data'][:4], dtype=torch.float32).reshape(4, 3, 32, 32)

        torch.manual_seed(42)
        np.random.seed(42)
        random.seed(42)

        enas = ENAS(inputShape=torch.Size([4, 3, 32, 32]))
        enas.construct()

        itr = enas.graph.getSampleArchitectures('input') 
        
        nextSample = next(itr)
        enas.sampleArchitecture(nextSample)
        enasOutput = enas.sample(tensorData)
        
        nextSample = next(itr)
        enas.sampleArchitecture(nextSample)
        enasOutput = enas.sample(tensorData)
        
        nextSample = next(itr)
        enas.sampleArchitecture(nextSample)
        enasOutput = enas.sample(tensorData)

        total = 100
        for i in tqdm(range(total), desc="Testing Forward Prop for Random Architectures"):
            sample = enas.graph.getRandomSampleArchitecture()
            enas.sampleArchitecture(sample)
            enasOutput = enas.sample(tensorData)


if __name__ == '__main__':
    unittest.main()
