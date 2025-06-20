import unittest
import os, io, sys, inspect
from unittest.mock import patch
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
graphdir = os.path.dirname(parentdir)
sys.path.insert(0, graphdir)
from classes.Nodes import *
from classes.Graph import Graph, GraphHandler
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

    
    def testBfs(self):
        """
        Tests the Graph class' bfs method..

        :param self: an instance of the graphTests class.
        """
        graph = GraphHandler()
        graph.construct(inputShape=torch.Size([4, 3, 32, 32]))

        exp = ['Input', 'No Normalization', 'Batch Normalization', '3x3 Conv(oc=4)', 
               '3x3 Conv(oc=8)', '3x3 Conv(oc=16)', '3x3 Conv(oc=32)', '5x5 Conv(oc=4)', 
               '5x5 Conv(oc=8)', '5x5 Conv(oc=16)', '5x5 Conv(oc=32)', 'No Normalization', 
               'Batch Normalization', 'No Activation', 'Relu Activation', 'No Pooling', 
               'Max Pooling', '3x3 Conv(oc=4)', '3x3 Conv(oc=8)', '3x3 Conv(oc=16)', 
               '3x3 Conv(oc=32)', '5x5 Conv(oc=4)', '5x5 Conv(oc=8)', '5x5 Conv(oc=16)', 
               '5x5 Conv(oc=32)', 'No Normalization', 'Batch Normalization', 'No Activation', 
               'Relu Activation', 'No Pooling', 'Max Pooling', 'Flatten', 'Linear(of=16)', 
               'Linear(of=32)', 'Linear(of=64)', 'Linear(of=128)', 'Linear(of=256)', 
               'No Activation', 'Relu Activation', 'Linear(of=16)', 'Linear(of=32)', 
               'Linear(of=64)', 'Linear(of=128)', 'Linear(of=256)', 'No Activation', 
               'Relu Activation', 'Linear(of=10)', 'No Activation', 'Relu Activation', 'Output']
        test = []
        for node in graph.bfs(graph.graph.graph['input']['node']):
            test.append(str(node))

        self.assertTrue(test == exp)
 
        
if __name__ == '__main__':
    unittest.main()
