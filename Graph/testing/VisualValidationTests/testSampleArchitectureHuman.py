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


    @patch('builtins.input', 
            side_effect=['0', '1', '1', '1', '0', '7', '1', '1', '1', '0', '3', '1', '1', '1', '0', '0', '0'])
    def testSampleArchitectureHuman(self, mock_input):
        """
        Tests the ability of the Graph class to sample NN architecture from 
        Graph via user input..
        
        :param self: An instance of the graphTests class.
        """
        sampleGraphPath = os.path.join(currentdir, 'TestFiles/sampleTestGraph.txt')
        graph = Graph()
        graph.construct(inputShape=torch.Size([4, 3, 32, 32]))

        # Sample Architecture by siming user input
        filepath = os.path.join(currentdir,'TestFiles', 'testSampleArchitectureHumanInput.txt')
        with open(filepath, 'w') as file:
            graph.sampleArchitectureHuman(clearTerminal=False, output=file)

        # Render the graph
        renderGraph(graph, os.path.join(currentdir, 'Temp'))

        
if __name__ == '__main__':
    unittest.main()
