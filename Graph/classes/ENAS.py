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
        self.sample = None


    def construct(self):
        self.graph.construct(inputShape=self.inputShape)
        self.mapPytorchLayers()
    

    def mapPytorchLayers(self):
        maxLinearSize = 0
        for curNode in self.graph.bfs(startNode=self.graph.graph['input']['node']):
            if curNode.pytorchLayerId is not None:
                pytorchLayerId = curNode.pytorchLayerId
                if curNode.pytorchLayerId not in self.pytorchLayers:
                    self.pytorchLayers[pytorchLayerId] = curNode.constructLayer()
                # Now set the shared layer in the node
                curNode.setSharedLayer(self.pytorchLayers[pytorchLayerId])

    
    def readGraph(self, filepath):
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"File not found: {filepath}")

        # Read the Graph file
        self.graph.readGraph(filepath)

        # Map Pytorch Layers
        self.mapPytorchLayers()


    def sampleArchitecture(self, sample):
        # Get the nodes from the graph based on the sample architecture
        nodes = self.graph.sampleArchitecture(sample)
        # Temp workaround because I'm tired
        out = torch.rand(self.inputShape)

        # I ain't heard no Fat Lady!
        layers = []

        for i in range(len(nodes)-1):
            layers.append(nodes[i].getLayer(out.shape))
            out = layers[-1](out)
         
        # Construct the CustomCNN instance with the nodes
        model = CustomCNN(layers, self.inputShape)
        self.sample = model


class CustomCNN(nn.Module):
    def __init__(self, layers, inputShape):
        super(CustomCNN, self).__init__()
        self.layers = layers


    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
