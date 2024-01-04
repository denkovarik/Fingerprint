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
     

if __name__ == '__main__':
    unittest.main()
