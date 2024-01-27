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

    
    def testPytorchInstalled(self):
        """
        Simple test for ensuring that Pytorch is installed.

        :param self: an instance of the graphTests class.
        """
        # Setup Linear Layer
        linearLayer = nn.Linear(in_features=5, out_features=2)
        weights = torch.tensor([[0.1788,  0.3492, -0.2402, -0.2631,  0.2751],
                                [0.0930,  0.0822,  0.3475,  0.1840,  0.1201]], requires_grad=True)
        linearLayer.weight = nn.Parameter(weights)
        # Make sure to also set the bias to zeros to match expected output
        linearLayer.bias.data.zero_()

        # Setup input and expected output
        inputTensor = torch.tensor([[-2.0942, -0.8275,  0.2748,  0.6571,  2.0056]])
        expOut = torch.tensor([[-0.3506,  0.1945]])

        # Run Test
        out = linearLayer(inputTensor)
        self.assertTrue(torch.allclose(out, expOut, atol=1e-3), "Output is not as expected")
 
        
if __name__ == '__main__':
    unittest.main()
