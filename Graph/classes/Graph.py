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

        
    def addActivationLayer(self):
        for act in self.activationOptions:
            nodeName = 'L' + str(self.layer) + '_' + act.value
            self.addNode(nodeType=NodeType.ACTIVATION, name=nodeName, activationType=act)
        self.prevNodes = self.curNodes
        self.curNodes = []

          
    def addConvolutionalLayer(self, layer, inputShape):
        maxNumChannels = max(self.ALLOWED_NUMBER_OF_CONVOLUTION_CHANNELS)
        outShapes = [inputShape]
        for kernel in self.ALLOWED_KERNEL_SIZES:
            pytorchLayerId = uuid.uuid4()
            for oc in self.ALLOWED_NUMBER_OF_CONVOLUTION_CHANNELS:
                nodeName = 'L' + str(self.layer) + '_' + str(kernel[0]) + 'x' \
                         + str(kernel[1]) + '_Conv(oc=' + str(oc) + ')'
                self.addNode(nodeType=NodeType.CONVOLUTION, 
                             name=nodeName, 
                             kernel_size=kernel, 
                             maxNumInputChannels=inputShape[1], 
                             maxNumOutputChannels=maxNumChannels, 
                             numOutputChannels=oc, layer=layer, 
                             pytorchLayerId=pytorchLayerId)

                outShape = SharedConv2d.calcOutSize(inputShape, oc, kernel)
                outShapes.append(outShape)
        self.prevNodes = self.curNodes
        self.curNodes = [] 
        maxOutShape = max(outShapes, key=lambda x: torch.tensor(x).prod().item())
        return maxOutShape

                         
    def addConvolutionalLayers(self, inputShape):
        outShape = inputShape
        outShapes = []
        for i in range(self.numConvLayers):       
            self.layer += 1
            outShape = self.addConvolutionalLayer(self.layer, inputShape)
            self.addNormalizationLayer()                
            self.addPoolingLayer()
            outShapes.append(outShape)
            inputShape = outShape
        return outShape

            
    def addFlattenLayer(self, inputShape):
        self.addNode(nodeType=NodeType.FLATTEN, name='L' + str(self.layer) + '_' + 'flatten')
        self.prevNodes = self.curNodes
        self.curNodes = []
        flattenedShape = torch.tensor(inputShape[1:]).prod().item()
        outShape = torch.Size([inputShape[0], flattenedShape])
        return outShape

        
    def addInputLayer(self, inputShape):
        self.addNode(nodeType=NodeType.INPUT, inputShape=inputShape)
        self.prevNodes = self.curNodes
        self.curNodes = []
        self.addNormalizationLayer()
        return inputShape

    
    def addLinearLayer(self, layer, pytorchLayerId, inputShape): 
        maxNumFeatures = max(self.ALLOWED_NUMBER_OF_LINEAR_FEATURES)
        for of in self.ALLOWED_NUMBER_OF_LINEAR_FEATURES:
            nodeName = 'L' + str(self.layer) + '_Linear(of=' + str(of) + ')' 
            self.addNode(nodeType=NodeType.LINEAR, 
                         name=nodeName, 
                         maxNumInFeatures=inputShape[1], 
                         maxNumOutFeatures=maxNumFeatures, 
                         numOutFeatures=of, layer=layer, 
                         pytorchLayerId=pytorchLayerId)
        self.prevNodes = self.curNodes
        self.curNodes = []
        return torch.Size([inputShape[0], maxNumFeatures])

     
    def addLinearLayers(self, inputShape):
        outShapes = []
        outShape = inputShape
        for i in range(self.numLinearLayers - 1):
            self.layer += 1            
            outShape = self.addLinearLayer(self.layer, uuid.uuid4(), outShape)
            self.addActivationLayer()                                                  
        self.layer += 1            
        nodeName = 'L' + str(self.layer) + '_Linear(of=' + str(10) + ')' 
        self.addNode(nodeType=NodeType.LINEAR, 
                     name=nodeName, 
                     maxNumInFeatures=outShape[1], 
                     maxNumOutFeatures=self.numClasses, 
                     numOutFeatures=self.numClasses, 
                     layer=self.layer, 
                     pytorchLayerId=uuid.uuid4())    
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

        
    def addOutputLayer(self):
        self.addNode(nodeType=NodeType.OUTPUT)

                
    def addPoolingLayer(self):
        for opt in self.poolingOptions:
            nodeName = 'L' + str(self.layer) + '_' + opt.value
            self.addNode(nodeType=NodeType.POOLING, name=nodeName, poolingType=opt)
        self.prevNodes = self.curNodes
        self.curNodes = []


    def bfs(self, startNode):
        """
        Generator function to perform bfs on the graph

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
       
 
    def construct(self, inputShape):
        self.layer = 0
        self.prevNodes = []
        self.curNodes = []   
        outShape = self.addInputLayer(inputShape)    
        outShape = self.addConvolutionalLayers(outShape)       
        outShape = self.addFlattenLayer(outShape)
        outShape = self.addLinearLayers(outShape)            
        self.addOutputLayer()


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
                print('|'.center(columns))
                print('V'.center(columns))
        print("")


    def readGraph(self, filepath):
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"File not found: {filepath}")

        # Read the Graph file
        with open(filepath, 'rb') as file:
            self.graph = pickle.load(file)
            
   
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


    def sampleArchitecture(self, sample):
        curNode = self.graph['input']
        self.sample = []
        ind = 0
 
        while curNode["node"].name != 'output':
            if ind >= len(sample):
                raise Exception("Output node could not be reached")
            curNode = self.graph[curNode["edges"][sample[ind]]]
            self.sample.append(sample[ind])
            ind += 1


    def sampleArchitectureHuman(self, clearTerminal=True, output=sys.stdout):
        original_stdout = sys.stdout
        sys.stdout = output
        if self.graph == {}:
            print('Please construct graph first')
            sys.stdout = original_stdout
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
        sys.stdout = original_stdout


    def writeGraph2File(self, filepath):
        # Ensure dir path exists
        dirpath = os.path.dirname(filepath)
        if dirpath and not os.path.exists(dirpath):
            os.makedirs(dirpath)

        with open(filepath, 'wb') as file:
            pickle.dump(self.graph, file)

