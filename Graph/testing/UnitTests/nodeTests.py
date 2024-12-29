import unittest
import os, io, sys, inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
graphdir = os.path.dirname(parentdir)
sys.path.insert(0, graphdir)
from classes.Nodes import *
from classes.Graph import *
import uuid
import torch
import torch.nn as nn
import numpy as np


class nodeTests(unittest.TestCase):
    """
    Runs all tests.
    """ 
        
    def testExecution(self):
        """
        Tests the ability of the allTests class to run a test.
        
        :param self: An instance of the allTests class.
        """
        self.assertTrue(True)


    def testNodeTypes(self):
        """
        Tests the ability to use NodeType enum
        
        :param self: An instance of the allTests class.
        """
        # INPUT
        nodeType = NodeType.INPUT
        self.assertTrue(nodeType == NodeType.INPUT)
        self.assertTrue(nodeType.value == 'input')

        # OUTPUT
        nodeType = NodeType.OUTPUT
        self.assertTrue(nodeType == NodeType.OUTPUT)
        self.assertTrue(nodeType.value == 'output')

        # CONVOLUTION
        nodeType = NodeType.CONVOLUTION
        self.assertTrue(nodeType == NodeType.CONVOLUTION)
        self.assertTrue(nodeType.value == 'convolution')

        # NORMALIZATION
        nodeType = NodeType.NORMALIZATION
        self.assertTrue(nodeType == NodeType.NORMALIZATION)
        self.assertTrue(nodeType.value == 'normalization')

        # POOLING
        nodeType = NodeType.POOLING
        self.assertTrue(nodeType == NodeType.POOLING)
        self.assertTrue(nodeType.value == 'pooling')

        # FLATTEN
        nodeType = NodeType.FLATTEN
        self.assertTrue(nodeType == NodeType.FLATTEN)
        self.assertTrue(nodeType.value == 'flatten')

        # LINEAR
        nodeType = NodeType.LINEAR
        self.assertTrue(nodeType == NodeType.LINEAR)
        self.assertTrue(nodeType.value == 'linear')

        # ACTIVATION
        nodeType = NodeType.ACTIVATION
        self.assertTrue(nodeType == NodeType.ACTIVATION)
        self.assertTrue(nodeType.value == 'activation')


    def testNormalizationTypes(self):
        """
        Tests the ability to use NormalizationType enum
        
        :param self: An instance of the allTests class.
        """
        # NO_NORM
        normType = NormalizationType.NO_NORM
        self.assertTrue(normType == NormalizationType.NO_NORM)
        self.assertTrue(normType.value == 'noNorm')

        # BATCH_NORM
        normType = NormalizationType.BATCH_NORM
        self.assertTrue(normType == NormalizationType.BATCH_NORM)
        self.assertTrue(normType.value == 'batchNorm')


    def testPoolingTypes(self):
        """
        Tests the ability to use PoolingType enum
        
        :param self: An instance of the allTests class.
        """
        # NO_POOLING
        poolType = PoolingType.NO_POOLING
        self.assertTrue(poolType == PoolingType.NO_POOLING)
        self.assertTrue(poolType.value == 'noPooling')

        # MAX_POOLING
        poolType = PoolingType.MAX_POOLING
        self.assertTrue(poolType == PoolingType.MAX_POOLING)
        self.assertTrue(poolType.value == 'maxPooling')


    def testActivationTypes(self):
        """
        Tests the ability to use ActivationType enum
        
        :param self: An instance of the allTests class.
        """
        # RELU
        actType = ActivationType.RELU
        self.assertTrue(actType == ActivationType.RELU)
        self.assertTrue(actType.value == 'reluActivation')
        
        # LINEAR
        actType = ActivationType.NONE
        self.assertTrue(actType == ActivationType.NONE)
        self.assertTrue(actType.value == 'noActivation')


    def testNode(self):
        """
        Tests the ability to construct and use the Node class
        
        :param self: An instance of the allTests class.
        """
        # Node
        node = Node("name", "displayName")
        self.assertTrue(node.name == "name")
        self.assertTrue(node.displayName == "displayName")
        self.assertTrue(str(node) == "displayName")
        self.assertTrue(node.__repr__() == "displayName")
    

    def testInputNode(self):
        """
        Tests the ability to construct and use the InputNode class
        
        :param self: An instance of the allTests class.
        """
        node = InputNode(inputShape=torch.Size([4, 3, 32, 32]))
        self.assertTrue(node.numChannels == 3)
        self.assertTrue(node.name == 'input')
        self.assertTrue(node.displayName == 'Input(numChannels=3)')
    

    def testOutputNode(self):
        """
        Tests the ability to construct and use the OutputNode class
        
        :param self: An instance of the allTests class.
        """
        node = OutputNode()
        self.assertTrue(node.name == 'output')
        self.assertTrue(node.displayName == 'Output')
    

    def testNormalizationNode(self):
        """
        Tests the ability to construct and use a node of the NormalizationNode 
        class
        
        :param self: An instance of the nodeTests class.
        """
        pytorchLayerId = uuid.uuid4()
        node = NormalizationNode(name="name", 
                                 normalizationType=NormalizationType.BATCH_NORM, 
                                 numFeatures=12, pytorchLayerId=pytorchLayerId)
        self.assertTrue(node.name == 'name')
        self.assertTrue(node.displayName == 'Batch Normalization')
        self.assertTrue(node.normalizationType == NormalizationType.BATCH_NORM)
    

    def testPoolingNode(self):
        """
        Tests the ability to construct and use a node of the PoolingNode 
        class
        
        :param self: An instance of the nodeTests class.
        """
        node = PoolingNode(name='name', poolingType=PoolingType.MAX_POOLING)
        self.assertTrue(node.name == 'name')
        self.assertTrue(node.displayName == 'Max Pooling')
        self.assertTrue(node.poolingType == PoolingType.MAX_POOLING)
    

    def testCovolutionalNode(self):
        """
        Tests the ability to construct and use a node of the CovolutionalNode 
        class
        
        :param self: An instance of the nodeTests class.
        """
        # Valid Construction with a tuple of 2 ints for kernel size
        pytorchLayerId = uuid.uuid4()
        node = ConvolutionalNode(name='name', kernel_size=(3,3), 
                                 maxNumInputChannels=128, 
                                 maxNumOutputChannels=128, 
                                 numOutputChannels=32,
                                 layer=0, pytorchLayerId=pytorchLayerId)
        self.assertTrue(node.name == 'name')
        self.assertTrue(node.layer == 0)
        self.assertTrue(node.pytorchLayerId == pytorchLayerId)
        self.assertTrue(node.displayName == '3x3 Conv(oc=32)')
        self.assertTrue(node.kernel_size == (3,3))
        self.assertTrue(node.maxNumInputChannels == 128)
        self.assertTrue(node.numOutputChannels == 32)

        # Valid Construction with a list of 2 ints for kernel size
        node = ConvolutionalNode(name='name', kernel_size=[3,3], 
                                 maxNumInputChannels=128,
                                 maxNumOutputChannels=128,
                                 numOutputChannels=32,
                                 layer=0, pytorchLayerId=pytorchLayerId)
        self.assertTrue(node.name == 'name')
        self.assertTrue(node.layer == 0)
        self.assertTrue(node.displayName == '3x3 Conv(oc=32)')
        self.assertTrue(node.kernel_size == (3,3))
        self.assertTrue(node.maxNumInputChannels == 128)
        self.assertTrue(node.numOutputChannels == 32)

        # Valid Construction with int for kernel size
        node = ConvolutionalNode(name='name', kernel_size=3, 
                                 maxNumInputChannels=128, 
                                 maxNumOutputChannels=128,
                                 numOutputChannels=32,
                                 layer=0, pytorchLayerId=pytorchLayerId)
        self.assertTrue(node.kernel_size == (3,3))
        
        # Invalid Construction with list of 3 ints for kernel size
        errMsg = 'Node of type ConvolutionNode constructed with invalid '
        errMsg += 'kernel size as a list of strings'
        testPassed = False
        try:
            node = ConvolutionalNode(name='name', kernel_size=[3,3,3], 
                                 maxNumInputChannels=128, numOutputChannels=32)
        except:
            testPassed = True

        self.assertTrue(testPassed, errMsg)
        
        # Invalid Construction with list of strings for kernel size
        errMsg = 'Node of type ConvolutionNode constructed with invalid '
        errMsg += 'kernel size as a list of strings'
        testPassed = False
        try:
            node = ConvolutionalNode(name='name', kernel_size=['3','3'], 
                                 maxNumInputChannels=128, numOutputChannels=32)
        except:
            testPassed = True

        self.assertTrue(testPassed, errMsg)
        
        # Invalid Construction with a string for kernel size
        testPassed = False
        try:
            node = ConvolutionalNode(name='name', kernel_size='3', 
                                 maxNumInputChannels=128, numOutputChannels=32)
        except:
            testPassed = True

        self.assertTrue(testPassed, errMsg)
    

    def testFlattenNode(self):
        """
        Tests the ability to construct and use a node of the FlattenNode 
        class
        
        :param self: An instance of the nodeTests class.
        """
        node = FlattenNode(name='name')
        self.assertTrue(node.name == 'name')
        self.assertTrue(node.displayName == 'Flatten')
    

    def testLinearNode(self):
        """
        Tests the ability to construct and use a node of the LinearNode 
        class
        
        :param self: An instance of the nodeTests class.
        """
        pytorchLayerId = uuid.uuid4()
        node = LinearNode(name='name', 
                          maxNumInFeatures=512, 
                          maxNumOutFeatures=512,
                          numOutFeatures=32, 
                          layer=1, pytorchLayerId=pytorchLayerId)
        self.assertTrue(node.name == 'name')
        self.assertTrue(node.layer == 1)
        self.assertTrue(node.pytorchLayerId == pytorchLayerId)
        self.assertTrue(node.maxNumInFeatures == 512)
        self.assertTrue(node.numOutFeatures == 32)
        self.assertTrue(node.displayName == 'Linear(of=32)')


    def testActivationNode(self):
        """
        Tests the ability to construct and use a node of the ActivationNode 
        class
        
        :param self: An instance of the nodeTests class.
        """
        node = ActivationNode(name='name', activationType=ActivationType.RELU)
        self.assertTrue(node.name == 'name')
        self.assertTrue(node.displayName == 'Relu Activation')
        self.assertTrue(node.activationType == ActivationType.RELU)


    def testNodeFactory(self):
        """
        Tests the NodeFactory class
        
        :param self: An instance of the nodeTests class.
        """
        nodeFactory = NodeFactory()
        self.assertTrue(isinstance(nodeFactory, NodeFactory)) 

        # Input Node
        node = nodeFactory.createNode(NodeType.INPUT, 
                                      inputShape=torch.Size([4, 1, 32, 32]))
        self.assertTrue(isinstance(node, InputNode))
        self.assertTrue(node.name == 'input')
        self.assertTrue(node.numChannels == 1)       
 
        # Output Node
        node = nodeFactory.createNode(NodeType.OUTPUT)
        self.assertTrue(isinstance(node, OutputNode))
        self.assertTrue(node.name == 'output')
        self.assertTrue(node.displayName == 'Output')
       
        # Normalization Node
        pytorchLayerId = uuid.uuid4()
        node = nodeFactory.createNode(NodeType.NORMALIZATION, 
                                      name='name', 
                                      normalizationType=NormalizationType.BATCH_NORM,
                                      numFeatures=8, pytorchLayerId=pytorchLayerId)
        self.assertTrue(isinstance(node, NormalizationNode))
        self.assertTrue(node.name == 'name')
        self.assertTrue(node.displayName == 'Batch Normalization')
        self.assertTrue(node.normalizationType == NormalizationType.BATCH_NORM)
       
        # Pooling Node
        node = nodeFactory.createNode(NodeType.POOLING, 
                                      name='name', 
                                      poolingType=PoolingType.MAX_POOLING)
        self.assertTrue(isinstance(node, PoolingNode))
        self.assertTrue(node.name == 'name')
        self.assertTrue(node.displayName == 'Max Pooling')
        self.assertTrue(node.poolingType == PoolingType.MAX_POOLING)
       
        # Convolution Node
        pytorchLayerId = uuid.uuid4()
        node = nodeFactory.createNode(NodeType.CONVOLUTION, name='name', 
                                      kernel_size=5, 
                                      maxNumInputChannels=128, 
                                      maxNumOutputChannels=128,
                                      numOutputChannels=32, 
                                      layer=2, pytorchLayerId=pytorchLayerId)
        self.assertTrue(isinstance(node, ConvolutionalNode))
        self.assertTrue(node.layer == 2)
        self.assertTrue(node.pytorchLayerId == pytorchLayerId)
        self.assertTrue(node.name == 'name')
        self.assertTrue(node.displayName == '5x5 Conv(oc=32)')
        self.assertTrue(node.kernel_size == (5,5))
        self.assertTrue(node.maxNumInputChannels == 128)
        self.assertTrue(node.numOutputChannels == 32)
       
        # Flatten Node
        node = nodeFactory.createNode(NodeType.FLATTEN, name='name')
        self.assertTrue(isinstance(node, FlattenNode))
        self.assertTrue(node.name == 'name')
        self.assertTrue(node.displayName == 'Flatten')
       
        # Linear Node
        node = nodeFactory.createNode(NodeType.LINEAR, name='name', 
                                      maxNumInFeatures=128,
                                      maxNumOutFeatures=128, 
                                      numOutFeatures=64,
                                      layer=0, pytorchLayerId=uuid.uuid4())
        self.assertTrue(isinstance(node, LinearNode))
        self.assertTrue(node.name == 'name')
        self.assertTrue(node.displayName == 'Linear(of=64)')
        self.assertTrue(node.maxNumInFeatures == 128)
        self.assertTrue(node.numOutFeatures == 64)
       
        # Activation Node
        node = nodeFactory.createNode(NodeType.ACTIVATION, name='name', activationType=ActivationType.RELU)
        self.assertTrue(isinstance(node, ActivationNode))
        self.assertTrue(node.name == 'name')
        self.assertTrue(node.displayName == 'Relu Activation')
        self.assertTrue(node.activationType == ActivationType.RELU)
    
        
if __name__ == '__main__':
    unittest.main()
