import unittest
import os, io, sys, inspect
from unittest.mock import patch
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
graphdir = os.path.dirname(parentdir)
sys.path.insert(0, graphdir)
from classes.SharedLinear import SharedLinear
from utils import *
import torch
import torch.nn as nn
    

class sharedLinearTests(unittest.TestCase):
    """
    Runs graph tests.
    """ 
        
    def testExecution(self):
        """
        Tests the ability of the graphTests class to run a test.
        
        :param self: An instance of the sharedLinearTests class.
        """
        self.assertTrue(True)


    def testConstruction(self):
        """
        Tests construction of the SharedLinear class

        :param self: An instance of the sharedLinearTests class.
        """
        sharedLinear = SharedLinear(max_in_features=32, max_out_features=32)


    def testPrint(self):
        """
        Tests the overloaded to string fucntion

        :param self: An instance of the sharedLinearTests class.
        """
        linearLayer = nn.Linear(in_features=10, out_features=5)
        sharedLinear = SharedLinear(max_in_features=32, max_out_features=32)
        exp = "SharedLinear(max_in_features=32, max_out_features=32)"
        self.assertTrue(str(sharedLinear) == exp)

 
        
if __name__ == '__main__':
    unittest.main()
