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


    def testConstruction(self):
        """
        Tests construction of the SharedLinear class

        :param self: An instance of the graphTests class.
        """
        sharedLinear = SharedLinear(maxInFeatures=32, maxOutFeatures=32)  
 
        
if __name__ == '__main__':
    unittest.main()
