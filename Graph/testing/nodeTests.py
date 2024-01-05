import unittest
import os, io, sys, inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)
from classes.Nodes import *
from classes.Graph import *


class allTests(unittest.TestCase):
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
        # NodeType
        nodeType = NodeType.INPUT
        self.assertTrue(nodeType == NodeType.INPUT)
        self.assertTrue(nodeType.value == 'input')


    def testNormalizationTypes(self):
        """
        Tests the ability to use NormalizationType enum
        
        :param self: An instance of the allTests class.
        """
        # NormalizationType
        normType = NormalizationType.NO_NORM
        self.assertTrue(normType == NormalizationType.NO_NORM)
        self.assertTrue(normType.value == 'noNorm')


    def testPoolingTypes(self):
        """
        Tests the ability to use PoolingType enum
        
        :param self: An instance of the allTests class.
        """
        # PoolingType
        poolType = PoolingType.MAX_POOLING
        self.assertTrue(poolType == PoolingType.MAX_POOLING)
        self.assertTrue(poolType.value == 'maxPooling')


    def testActivationTypes(self):
        """
        Tests the ability to use ActivationType enum
        
        :param self: An instance of the allTests class.
        """
        # ActivationType
        actType = ActivationType.LINEAR
        self.assertTrue(actType == ActivationType.LINEAR)
        self.assertTrue(actType.value == 'linearActivation')


    def testNode(self):
        """
        Tests the ability to construct and use the Node class
        
        :param self: An instance of the allTests class.
        """
        # Node
        node = Node("name", "displayName")
        self.assertTrue(node.name == "name")
        self.assertTrue(node.displayName == "displayName")
    

    def testInputNode(self):
        """
        Tests the ability to construct and use the InputNode class
        
        :param self: An instance of the allTests class.
        """
        # Input Node
        node = InputNode(numChannels=3)
        self.assertTrue(node.numChannels == 3)
     

if __name__ == '__main__':
    unittest.main()
