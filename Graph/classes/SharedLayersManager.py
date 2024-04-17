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


class SharedLayersManager:
    def __init__(self):
        pass
    

    def mapPytorchLayers(self):
        maxLinearSize = 0
        for curNode in self.bfs(startNode=self.graph['input']['node']):
            if curNode.pytorchLayerId is not None:
                pytorchLayerId = curNode.pytorchLayerId 
                self.pytorchLayers[pytorchLayerId] = curNode.getPytorchLayer()
