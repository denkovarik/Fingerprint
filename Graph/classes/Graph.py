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


class Graph:
    def __init__(self):
        self.ALLOWED_KERNEL_SIZES = {(3, 3), (5, 5)}
        self.ALLOWED_NUMBER_OF_CONVOLUTION_CHANNELS = [4, 8, 16, 32]
        self.ALLOWED_NUMBER_OF_LINEAR_FEATURES = [16, 32, 64, 128, 256]
        self.graph = {}
        self.normalizationOptions = [NormalizationType.NO_NORM, NormalizationType.BATCH_NORM]    
        self.poolingOptions = [PoolingType.NO_POOLING, PoolingType.MAX_POOLING]    
        self.activationOptions = [ActivationType.LINEAR, ActivationType.RELU]
        self.nodeFactory = NodeFactory()
        self.numConvLayers = 2
        self.numLinearLayers = 3
        self.numClasses = 10
        self.prevNodes = []
        self.curNodes = []
        self.layer = 0
        self.sample = []
        self.pytorchLayers = {}
       
 
    def construct(self):
        self.layer = 0
        self.prevNodes = []
        self.curNodes = []   
        self.addInputLayer()    
        self.addConvolutionalLayers()       
        self.addFlattenLayer()
        self.addLinearLayers()            
        self.addOutputLayer()
        self.mapPytorchLayers()

        
    def addActivationLayer(self):
        for act in self.activationOptions:
            nodeName = 'L' + str(self.layer) + '_' + act.value
            self.addNode(nodeType=NodeType.ACTIVATION, name=nodeName, activationType=act)
        self.prevNodes = self.curNodes
        self.curNodes = []

          
    def addConvolutionalLayer(self, layer):
        maxNumInputChannels = max(self.ALLOWED_NUMBER_OF_CONVOLUTION_CHANNELS)
        for kernel in self.ALLOWED_KERNEL_SIZES:
            conv2dId = uuid.uuid4()
            for oc in self.ALLOWED_NUMBER_OF_CONVOLUTION_CHANNELS:
                nodeName = 'L' + str(self.layer) + '_' + str(kernel[0]) + 'x' \
                         + str(kernel[1]) + '_Conv(oc=' + str(oc) + ')'
                self.addNode(nodeType=NodeType.CONVOLUTION, 
                             name=nodeName, 
                             kernelSize=kernel, 
                             maxNumInputChannels=maxNumInputChannels, 
                             numOutputChannels=oc, layer=layer, 
                             conv2dId=conv2dId)
        self.prevNodes = self.curNodes
        self.curNodes = []

                         
    def addConvolutionalLayers(self):
        for i in range(self.numConvLayers):       
            self.layer += 1
            self.addConvolutionalLayer(self.layer)                                        
            self.addNormalizationLayer()                
            self.addPoolingLayer()

            
    def addFlattenLayer(self):
        self.addNode(nodeType=NodeType.FLATTEN, name='L' + str(self.layer) + '_' + 'flatten')
        self.prevNodes = self.curNodes
        self.curNodes = []

        
    def addInputLayer(self):
        self.addNode(nodeType=NodeType.INPUT, numChannels=3)
        self.prevNodes = self.curNodes
        self.curNodes = []
        self.addNormalizationLayer()

        
    def addOutputLayer(self):
        self.addNode(nodeType=NodeType.OUTPUT)

    
    def addLinearLayer(self, layer, linearId): 
        maxNumInputFeatures = max(self.ALLOWED_NUMBER_OF_LINEAR_FEATURES)
        for of in self.ALLOWED_NUMBER_OF_LINEAR_FEATURES:
            nodeName = 'L' + str(self.layer) + '_Linear(of=' + str(of) + ')' 
            self.addNode(nodeType=NodeType.LINEAR, 
                         name=nodeName, 
                         maxNumInFeatures=maxNumInputFeatures, 
                         numOutFeatures=of, layer=layer, linearId=linearId)
        self.prevNodes = self.curNodes
        self.curNodes = []

     
    def addLinearLayers(self):
        for i in range(self.numLinearLayers - 1):
            self.layer += 1            
            self.addLinearLayer(self.layer, uuid.uuid4())
            self.addActivationLayer()                                                  
        self.layer += 1            
        nodeName = 'L' + str(self.layer) + '_Linear(of=' + str(10) + ')' 
        self.addNode(nodeType=NodeType.LINEAR, 
                     name=nodeName, 
                     maxNumInFeatures=max(self.ALLOWED_NUMBER_OF_LINEAR_FEATURES), 
                     numOutFeatures=self.numClasses, layer=self.layer, linearId=uuid.uuid4())    
        self.prevNodes = self.curNodes
        self.curNodes = []
        self.addActivationLayer()

            
    def addNode(self, **kwargs):
        node = self.nodeFactory.createNode(**kwargs)      
        self.graph[node.name] = {'node': node, 'edges': []}
        for prev in self.prevNodes:
            prevNode = self.graph[prev]
            prevNode['edges'].append(node.name)
        self.curNodes.append(node.name)  

        
    def addNormalizationLayer(self):
        for opt in self.normalizationOptions:
            nodeName = 'L' + str(self.layer) + '_' + opt.value
            self.addNode(nodeType=NodeType.NORMALIZATION, name=nodeName, normalizationType=opt)
        self.prevNodes = self.curNodes
        self.curNodes = []         

                
    def addPoolingLayer(self):
        for opt in self.poolingOptions:
            nodeName = 'L' + str(self.layer) + '_' + opt.value
            self.addNode(nodeType=NodeType.POOLING, name=nodeName, poolingType=opt)
        self.prevNodes = self.curNodes
        self.curNodes = []


    def printSampleArchitecture(self, sample):
        # Get the terminal size
        columns, rows = shutil.get_terminal_size()
        curNode = self.graph['input']
        # Print Input Node
        centeredName = curNode['node'].displayName.center(columns)
        print(centeredName)
        print('|'.center(columns))
        print('V'.center(columns))
        for i in range(len(sample)):
            curNode = self.graph[self.graph[curNode['node'].name]['edges'][sample[i]]]
            # Center the node display name
            centeredName = curNode['node'].displayName.center(columns)
            print(centeredName)
            if i + 1 < len(sample):
                # Center the '|' character
                print('|'.center(columns))
                # Center the 'V' character
                print('V'.center(columns))
        print("")


    def readGraph(self, filepath):
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"File not found: {filepath}")

        # Read the Graph file
        with open(filepath, 'rb') as file:
            self.graph = pickle.load(file)

        # Map Pytorch Layers
        #self.mapPytorchLayers()


    def bfs(self, startNode):
        """
        Generator function to performa bfs on the graph

        :param self: An instance of the Graph class
        :param startNode: Start node for the bfs
        """
        nextNodes = queue.Queue()
        visited = set()
        nextNodes.put(startNode)
        visited.add(startNode)

        while not nextNodes.empty():
            cur = nextNodes.get()
            yield cur

            for edge in self.graph[cur.name]['edges']:
                edjNode = self.graph[edge]['node']
                if edjNode not in visited:
                    nextNodes.put(edjNode)
                    visited.add(edjNode)


    def mapPytorchLayers(self):
        for curNode in self.bfs(startNode=self.graph['input']['node']):
            if isinstance(curNode, ConvolutionalNode): 
                conv2dId = curNode.conv2dId 
                if conv2dId not in self.pytorchLayers.keys():
                    self.pytorchLayers[conv2dId] = {'Type': 'Conv2d',
                                                    'id': conv2dId,
                                                    'layerNum': curNode.layer, 
                                                    'KernelSize': curNode.kernelSize,
                                                    'InputChannels': curNode.maxNumInputChannels,
                                                    'OutputChannels': curNode.numOutputChannels}
                else:
                    oc = max(self.pytorchLayers[conv2dId]['OutputChannels'], curNode.numOutputChannels)
                    self.pytorchLayers[conv2dId]['OutputChannels'] = oc
            elif isinstance(curNode, LinearNode):
                linearId = curNode.linearId
                if linearId not in self.pytorchLayers.keys():
                    self.pytorchLayers[linearId] = {'Type': 'Linear',
                                                    'id': linearId,
                                                    'layerNum': curNode.layer, 
                                                    'InputChannels': curNode.maxNumInFeatures,
                                                    'OutputChannels': curNode.numOutFeatures}
                else:
                    oc = max(self.pytorchLayers[linearId]['OutputChannels'], curNode.numOutFeatures)
                    self.pytorchLayers[conv2dId]['OutputChannels'] = oc
        
        for key in self.pytorchLayers.keys():
            if self.pytorchLayers[key]['Type'] == 'Conv2d':
                ic = self.pytorchLayers[key]['OutputChannels']
                oc = self.pytorchLayers[key]['InputChannels']
                ks = self.pytorchLayers[key]['KernelSize']
                self.pytorchLayers[key]['Layer'] = nn.Conv2d(ic, oc, kernel_size=ks)
            elif self.pytorchLayers[key]['Type'] == 'Linear':
                ic = self.pytorchLayers[key]['OutputChannels']
                oc = self.pytorchLayers[key]['InputChannels']
                self.pytorchLayers[key]['Layer'] = nn.Linear(ic, oc)
            

              
    def render(self, dirPath=os.path.join(parentdir, "Graphs/GraphVisualizations/")):
        nodes = []
        edges = [] 
        if len(self.sample) > 0:
            curNode = self.graph['input']
            nodes.append(curNode['node'].name)
            for edge in self.sample:
                edges.append(self.graph[curNode['node'].name]['edges'][edge])
                curNode = self.graph[self.graph[curNode['node'].name]['edges'][edge]]
                nodes.append(curNode['node'].name)

        # Initialize the graph
        g = Digraph('G', filename = os.path.join(dirPath + 'enas_network_search_space'))

        # Define graph attributes
        g.attr(rankdir='TB')  # 'TB' for top-to-bottom graph, 'LR' for left-to-right
        g.attr('node', shape='box', style='filled', color='lightgrey')
        g.attr(ranksep='2.0')  # Increase the space between layers, adjust the value as needed
        
        # Add Nodes
        for val in self.graph.values():
            node = val['node']
            if node.name in nodes:
                g.node(node.name, node.displayName, color='green', style='bold')
            else:
                g.node(node.name, node.displayName)
            
        # Add Edges
        for val in self.graph.values():
            node = val['node']
            for edge in val['edges']:
                if node.name in nodes and edge in edges:
                    g.edge(node.name, edge, color='green', style='bold')
                else:
                    g.edge(node.name, edge)
                
        # Specify the output format and render the graph
        g.format = 'png'
        filePath = os.path.join(dirPath, 'enas_network_search_space_visualization')
        g.render(filePath)

        return filePath + '.png'


    def sampleArchitectureHuman(self, clearTerminal=True):
        if self.graph == {}:
            print('Please construct graph first')
            return
       
        curNode = self.graph['input']
        self.sample = []
 
        while curNode["node"].name != 'output':
            if clearTerminal:
                clearScreen()
            self.printSampleArchitecture(self.sample)
            print("Select the next node from the following options:\n")
            num = 0
            for edge in curNode["edges"]:
                print(str(num) + ': ' + edge)
                num += 1
            validSelection = False
            while not validSelection:
                try:
                    selected = input("\nSelect Node: ")
                    selectedInt = int(selected)
                    if selectedInt < 0 or selectedInt > len(curNode["edges"]):
                        raise ValueError
                    validSelection = True
                except ValueError:
                    print("Please enter a value from 0 to " + str(len(curNode["edges"])-1))
            
            print("selectedInt: ", end="")
            print(selectedInt)
            curNode = self.graph[curNode["edges"][selectedInt]]
            self.sample.append(selectedInt)
        print("")
        self.printSampleArchitecture(self.sample)


    def writeGraph2File(self, filepath):
        # Assuming your_dict is the dictionary you want to save
        with open(filepath, 'wb') as file:
            pickle.dump(self.graph, file)

