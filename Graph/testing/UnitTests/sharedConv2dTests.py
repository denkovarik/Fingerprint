import unittest
import os, io, sys, inspect
from unittest.mock import patch
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
graphdir = os.path.dirname(parentdir)
sys.path.insert(0, graphdir)
from classes.SharedConv2d import SharedConv2d
from utils import *
import torch
import torch.nn as nn
import random
import numpy as np


class sharedConv2DTests(unittest.TestCase):
    """
    Runs graph tests.
    """ 
        
    def testExecution(self):
        """
        Tests the ability of the sharedConv2DTests class to run a test.
        
        :param self: An instance of the sharedConv2DTests class.
        """
        self.assertTrue(True)


    def testConstruction(self):
        """
        Tests construction of the SharedConv2DTests class

        :param self: An instance of the sharedConv2DTests class.
        """
        sharedConv = SharedConv2d(kernel_size=3, in_channels=32, out_channels=32)
        self.assertTrue(isinstance(sharedConv, SharedConv2d))


    def testCalcOutputSize(self):
        """
        Tests method calcOutSize of the SharedConv2DTests class

        :param self: An instance of the sharedConv2DTests class.
        """
        # Get test batch
        testBatchPath = os.path.join(currentdir, 'TestFiles/cifar10_test_batch_pickle')
        self.assertTrue(testBatchPath)
        testBatch = unpickle(testBatchPath)
        imgData = testBatch[b'data'][:4]
        batch = imgData.reshape(4, 3, 32, 32)
        tensorData = torch.tensor(batch, dtype=torch.float32)
        tensorData = torch.tensor(testBatch[b'data'][:4], dtype=torch.float32).reshape(4, 3, 32, 32)
        
        # Test 1: Kernel Size 3x3
        conv2d = nn.Conv2d(3, 8,kernel_size=3)
        sharedConv2d = SharedConv2d(3, 8, kernel_size=3)
        # Calculate what the output size should be
        calcOutputSize = sharedConv2d.calcOutSize(inputHeight=tensorData.shape[2],
                                                  inputWidth=tensorData.shape[3])
        # Forward prop
        outConv2d = conv2d(tensorData)
        outSharedConv2d = sharedConv2d(tensorData, 3, 8)
        self.assertTrue(calcOutputSize[0] == outConv2d.shape[2])
        self.assertTrue(calcOutputSize[1] == outConv2d.shape[3])
        self.assertTrue(calcOutputSize[0] == outSharedConv2d.shape[2])
        self.assertTrue(calcOutputSize[1] == outSharedConv2d.shape[3])

        # Test 2: Kernel Size 5x5
        conv2d = nn.Conv2d(3, 16, kernel_size=5)
        sharedConv2d = SharedConv2d(3, 16, kernel_size=5)
        # Calculate what the output size should be
        calcOutputSize = sharedConv2d.calcOutSize(inputHeight=tensorData.shape[2],
                                                  inputWidth=tensorData.shape[3])
        # Forward prop
        outConv2d = conv2d(tensorData)
        outSharedConv2d = sharedConv2d(tensorData, 3, 16)
        self.assertTrue(calcOutputSize[0] == outConv2d.shape[2])
        self.assertTrue(calcOutputSize[1] == outConv2d.shape[3])
        self.assertTrue(calcOutputSize[0] == outSharedConv2d.shape[2])
        self.assertTrue(calcOutputSize[1] == outSharedConv2d.shape[3])

        # Test 3: Shared Conv2d ouput number of Channels less than max 
        #         Kernel Size 3x3
        conv2d = nn.Conv2d(3, 8,kernel_size=3)
        sharedConv2d = SharedConv2d(3, 16, kernel_size=3)
        # Calculate what the output size should be
        calcOutputSize = sharedConv2d.calcOutSize(inputHeight=tensorData.shape[2],
                                                  inputWidth=tensorData.shape[3])
        # Forward prop
        outConv2d = conv2d(tensorData)
        outSharedConv2d = sharedConv2d(tensorData, 3, 8)
        self.assertTrue(calcOutputSize[0] == outConv2d.shape[2])
        self.assertTrue(calcOutputSize[1] == outConv2d.shape[3])
        self.assertTrue(calcOutputSize[0] == outSharedConv2d.shape[2])
        self.assertTrue(calcOutputSize[1] == outSharedConv2d.shape[3])

        # Test 4: Shared Conv2d ouput number of Channels less than Conv2d 
        #         output channels Kernel Size 5x5
        conv2d = nn.Conv2d(3, 64,kernel_size=5)
        sharedConv2d = SharedConv2d(3, 256, kernel_size=5)
        # Calculate what the output size should be
        calcOutputSize = sharedConv2d.calcOutSize(inputHeight=tensorData.shape[2],
                                                  inputWidth=tensorData.shape[3])
        # Forward prop
        outConv2d = conv2d(tensorData)
        outSharedConv2d = sharedConv2d(tensorData, 3, 8)
        self.assertTrue(calcOutputSize[0] == outConv2d.shape[2])
        self.assertTrue(calcOutputSize[1] == outConv2d.shape[3])
        self.assertTrue(calcOutputSize[0] == outSharedConv2d.shape[2])
        self.assertTrue(calcOutputSize[1] == outSharedConv2d.shape[3])


    def testForwardPass(self):
        """
        Tests the forward pass for the SharedConv2DTests class.

        :param self: An instance of the SharedConv2DTests class.
        """
        torch.manual_seed(42)
        np.random.seed(42)
        random.seed(42)
        
        weights = torch.nn.init.kaiming_uniform_(torch.empty(8, 3, 3, 3), mode='fan_in', nonlinearity='relu')
        weightsSub = weights[:4].clone()

        # Construct control conv2d layer
        conv2d = nn.Conv2d(3, 4, 3)
        conv2d.weight.data = weightsSub
        conv2d.bias.data.zero_()
        # Construct shared conv2d layer
        sharedConv2d = SharedConv2d(3, 8, 3)
        sharedConv2d.weight = nn.Parameter(weights)
        sharedConv2d.bias.data.zero_()
        # Make sure initialization was done correctly
        self.assertTrue(conv2d.kernel_size == (3, 3))
        self.assertTrue(sharedConv2d.kernelSize == (3, 3))
        self.assertTrue(torch.allclose(conv2d.weight, weightsSub))
        self.assertTrue(torch.allclose(sharedConv2d.weight, weights))
        self.assertTrue(torch.all(conv2d.bias.eq(0)))
        self.assertTrue(torch.all(sharedConv2d.bias.eq(0)))
        self.assertTrue(torch.allclose(conv2d.weight, sharedConv2d.weight[:4]))
        # Get test batch
        testBatchPath = os.path.join(currentdir, 'TestFiles/cifar10_test_batch_pickle')
        self.assertTrue(testBatchPath)
        testBatch = unpickle(testBatchPath)
        imgData = testBatch[b'data'][:4]
        batch = imgData.reshape(4, 3, 32, 32)
        tensorData = torch.tensor(batch, dtype=torch.float32)
        tensorData = torch.tensor(testBatch[b'data'][:4], dtype=torch.float32).reshape(4, 3, 32, 32)
        # Forward prop
        outConv2d = conv2d(tensorData)
        outSharedConv2d = sharedConv2d(tensorData, 3, 4)
        # Test the output is the same
        self.assertTrue(torch.allclose(outConv2d, outSharedConv2d))

        
if __name__ == '__main__':
    unittest.main()
