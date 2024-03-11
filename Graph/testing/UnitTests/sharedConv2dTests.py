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


    def testCalOutputSize(self):
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
        calcOutputSize = SharedConv2d.calcOutSize(tensorData.shape, out_channels=8, kernel_size=3)
        # Forward prop
        outConv2d = conv2d(tensorData)
        outSharedConv2d = sharedConv2d(tensorData, 3, 8)
        self.assertTrue(calcOutputSize == outConv2d.shape)
        self.assertTrue(calcOutputSize == outSharedConv2d.shape)

        # Test 2: Kernel Size 5x5
        conv2d = nn.Conv2d(3, 16, kernel_size=5)
        sharedConv2d = SharedConv2d(5, 16, kernel_size=5)
        # Calculate what the output size should be
        calcOutputSize = SharedConv2d.calcOutSize(tensorData.shape, out_channels=16, kernel_size=5)
        # Forward prop
        outConv2d = conv2d(tensorData)
        outSharedConv2d = sharedConv2d(tensorData, 3, 16)
        self.assertTrue(calcOutputSize == outConv2d.shape)
        self.assertTrue(calcOutputSize == outSharedConv2d.shape)

        # Test 3: Shared Conv2d ouput number of Channels less than max 
        #         Kernel Size 3x3
        conv2d = nn.Conv2d(3, 8,kernel_size=3)
        sharedConv2d = SharedConv2d(3, 16, kernel_size=3)
        # Calculate what the output size should be
        calcOutputSize = SharedConv2d.calcOutSize(tensorData.shape, out_channels=8, kernel_size=3)
        # Forward prop
        outConv2d = conv2d(tensorData)
        outSharedConv2d = sharedConv2d(tensorData, 3, 8)
        self.assertTrue(calcOutputSize == outConv2d.shape)
        self.assertTrue(calcOutputSize == outSharedConv2d.shape)

        # Test 5: Calculating tensor dimensions for multiple runs
        # Conv2d
        conv2d1 = nn.Conv2d(3, 8, kernel_size=3)
        conv2d2 = nn.Conv2d(8, 16, kernel_size=5)
        conv2d3 = nn.Conv2d(16, 32, kernel_size=5)
        # Forward prop Conv2d Layers
        outConv2d1 = conv2d1(tensorData)
        outConv2d2 = conv2d2(outConv2d1)
        outConv2d3 = conv2d3(outConv2d2)
        # SharedConv2d
        sharedConv2d1 = SharedConv2d(3, 256, kernel_size=3) 
        sharedConv2d2 = SharedConv2d(256, 256, kernel_size=5) 
        sharedConv2d3 = SharedConv2d(256, 256, kernel_size=5)
        # Calculated SharedConv2d Shapes
        calcOutShape1 = SharedConv2d.calcOutSize(tensorData.shape, out_channels=8, kernel_size=3)
        calcOutShape2 = SharedConv2d.calcOutSize(calcOutShape1, out_channels=16, kernel_size=5)
        calcOutShape3 = SharedConv2d.calcOutSize(calcOutShape2, out_channels=32, kernel_size=5)
        # Forward Prop Shared Conv2d Layers
        outSharedConv2d1 = sharedConv2d1(tensorData, 3, 8)
        outSharedConv2d2 = sharedConv2d2(outSharedConv2d1, 8, 16)
        outSharedConv2d3 = sharedConv2d3(outSharedConv2d2, 16, 32)
        # Validate Shapes
        self.assertTrue(calcOutShape1 == outConv2d1.shape)
        self.assertTrue(calcOutShape1 == outSharedConv2d1.shape)
        self.assertTrue(calcOutShape2 == outConv2d2.shape)
        self.assertTrue(calcOutShape2 == outSharedConv2d2.shape)
        self.assertTrue(calcOutShape3 == outConv2d3.shape)
        self.assertTrue(calcOutShape3 == outSharedConv2d3.shape)


    def testGetOutputSize(self):
        """
        Tests method getOutSize of the SharedConv2DTests class

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
        calcOutputSize = sharedConv2d.getOutSize(tensorData.shape, out_channels=8)
        # Forward prop
        outConv2d = conv2d(tensorData)
        outSharedConv2d = sharedConv2d(tensorData, 3, 8)
        self.assertTrue(calcOutputSize == outConv2d.shape)
        self.assertTrue(calcOutputSize == outSharedConv2d.shape)

        # Test 2: Kernel Size 5x5
        conv2d = nn.Conv2d(3, 16, kernel_size=5)
        sharedConv2d = SharedConv2d(3, 16, kernel_size=5)
        # Calculate what the output size should be
        calcOutputSize = sharedConv2d.getOutSize(tensorData.shape, out_channels=16)
        # Forward prop
        outConv2d = conv2d(tensorData)
        outSharedConv2d = sharedConv2d(tensorData, 3, 16)
        self.assertTrue(calcOutputSize == outConv2d.shape)
        self.assertTrue(calcOutputSize == outSharedConv2d.shape)

        # Test 3: Shared Conv2d ouput number of Channels less than max 
        #         Kernel Size 3x3
        conv2d = nn.Conv2d(3, 8,kernel_size=3)
        sharedConv2d = SharedConv2d(3, 16, kernel_size=3)
        # Calculate what the output size should be
        calcOutputSize = sharedConv2d.getOutSize(tensorData.shape, out_channels=8)
        # Forward prop
        outConv2d = conv2d(tensorData)
        outSharedConv2d = sharedConv2d(tensorData, 3, 8)
        self.assertTrue(calcOutputSize == outConv2d.shape)
        self.assertTrue(calcOutputSize == outSharedConv2d.shape)

        # Test 5: Calculating tensor dimensions for multiple runs
        # Conv2d
        conv2d1 = nn.Conv2d(3, 8, kernel_size=3)
        conv2d2 = nn.Conv2d(8, 16, kernel_size=5)
        conv2d3 = nn.Conv2d(16, 32, kernel_size=5)
        # Forward prop Conv2d Layers
        outConv2d1 = conv2d1(tensorData)
        outConv2d2 = conv2d2(outConv2d1)
        outConv2d3 = conv2d3(outConv2d2)
        # SharedConv2d
        sharedConv2d1 = SharedConv2d(3, 256, kernel_size=3) 
        sharedConv2d2 = SharedConv2d(256, 256, kernel_size=5) 
        sharedConv2d3 = SharedConv2d(256, 256, kernel_size=5)
        # Calculated SharedConv2d Shapes
        calcOutShape1 = sharedConv2d1.getOutSize(tensorData.shape, out_channels=8)
        calcOutShape2 = sharedConv2d2.getOutSize(calcOutShape1, out_channels=16)
        calcOutShape3 = sharedConv2d3.getOutSize(calcOutShape2, out_channels=32)
        # Forward Prop Shared Conv2d Layers
        outSharedConv2d1 = sharedConv2d1(tensorData, 3, 8)
        outSharedConv2d2 = sharedConv2d2(outSharedConv2d1, 8, 16)
        outSharedConv2d3 = sharedConv2d3(outSharedConv2d2, 16, 32)
        # Validate Shapes
        self.assertTrue(calcOutShape1 == outConv2d1.shape)
        self.assertTrue(calcOutShape1 == outSharedConv2d1.shape)
        self.assertTrue(calcOutShape2 == outConv2d2.shape)
        self.assertTrue(calcOutShape2 == outSharedConv2d2.shape)
        self.assertTrue(calcOutShape3 == outConv2d3.shape)
        self.assertTrue(calcOutShape3 == outSharedConv2d3.shape)


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


    def testCheckDilation(self):
        """
        Tests the checkDilation static method for the SharedConv2DTests class.

        :param self: An instance of the SharedConv2DTests class.
        """
        intVal = 2
        tupleOfInts = (2, 2)
        self.assertTrue(SharedConv2d.checkDilation(intVal) == tupleOfInts)
        self.assertTrue(SharedConv2d.checkDilation(tupleOfInts) == tupleOfInts)


    def testCheckKernelSize(self):
        """
        Tests the checkKernelSize static method for the SharedConv2DTests class.

        :param self: An instance of the SharedConv2DTests class.
        """
        intVal = 2
        tupleOfInts = (2, 2)
        self.assertTrue(SharedConv2d.checkKernelSize(intVal) == tupleOfInts)
        self.assertTrue(SharedConv2d.checkKernelSize(tupleOfInts) == tupleOfInts)


    def testCheckPadding(self):
        """
        Tests the checkPadding static method for the SharedConv2DTests class.

        :param self: An instance of the SharedConv2DTests class.
        """
        intVal = 2
        tupleOfInts = (2, 2)
        self.assertTrue(SharedConv2d.checkPadding(intVal) == tupleOfInts)
        self.assertTrue(SharedConv2d.checkPadding(tupleOfInts) == tupleOfInts)


    def testCheckStride(self):
        """
        Tests the checkStride static method for the SharedConv2DTests class.

        :param self: An instance of the SharedConv2DTests class.
        """
        intVal = 2
        tupleOfInts = (2, 2)
        self.assertTrue(SharedConv2d.checkStride(intVal) == tupleOfInts)
        self.assertTrue(SharedConv2d.checkStride(tupleOfInts) == tupleOfInts)


    def testCnvrtInt2Tuple(self):
        """
        Tests the cnvertInt2Tuple static method for the SharedConv2DTests class.

        :param self: An instance of the SharedConv2DTests class.
        """
        intVal = 2
        tupleOfInts = (2, 2)
        self.assertTrue(SharedConv2d.cnvrtInt2Tuple(intVal) == tupleOfInts)


    def testIsTupleOfInts(self):
        """
        Tests the isTupleOfInts static method for the SharedConv2DTests class.

        :param self: An instance of the SharedConv2DTests class.
        """
        intVal = 2
        tupleOfInts = (2, 2)
        tupleOfStrs = ("2", "2")
        listOfInts = [2,2]

        self.assertTrue(SharedConv2d.isTupleOfInts(tupleOfInts))
        self.assertFalse(SharedConv2d.isTupleOfInts(tupleOfStrs))
        self.assertFalse(SharedConv2d.isTupleOfInts(intVal))
        self.assertFalse(SharedConv2d.isTupleOfInts(listOfInts))


    def testPrint(self):
        """
        Tests the overloaded to string fucntion

        :param self: An instance of the sharedLinearTests class.
        """
        conv2dLayer = nn.Conv2d(3, 8, kernel_size=3)
        sharedConv2d = SharedConv2d(in_channels=3, out_channels=8, kernel_size=3)
        exp = "SharedConv2d(3, 8, kernel_size=(3, 3))"
        self.assertTrue(str(sharedConv2d) == exp)

        
if __name__ == '__main__':
    unittest.main()
