import os, io, sys, inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
from graphviz import Digraph
from IPython.display import display, Image
import copy
from classes.Nodes import *
from utils import *
import pickle
import shutil
import uuid
import queue
import torch
import torch.nn as nn
from utils import ensureFilepathExists
from classes.SharedConv2d import SharedConv2d
from classes.SharedLinear import SharedLinear
from classes.Graph import Graph


class ENAS:
    def __init__(self, inputShape):
        self.graph = Graph()
        self.inputShape = inputShape
        self.pytorchLayers = {}


    def construct(self, inputShape):
        self.graph.construct(inputShape=inputShape)
        self.mapPytorchLayers()
    

    def mapPytorchLayers(self):
        maxLinearSize = 0
        for curNode in self.graph.bfs(startNode=self.graph.graph['input']['node']):
            if curNode.pytorchLayerId is not None:
                pytorchLayerId = curNode.pytorchLayerId 
                self.pytorchLayers[pytorchLayerId] = curNode.getPytorchLayer()

    
    def readGraph(self, filepath):
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"File not found: {filepath}")

        # Read the Graph file
        self.graph.readGraph(filepath)

        # Map Pytorch Layers
        self.mapPytorchLayers()

