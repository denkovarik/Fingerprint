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
 
        
if __name__ == '__main__':
    unittest.main()
