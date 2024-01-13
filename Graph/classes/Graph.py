import os, io, sys, inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
from graphviz import Digraph
from IPython.display import display, Image
import copy
from classes.Nodes import *
from utils import *
import pickle


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
       
 
    def construct(self):
        self.layer = 0
        self.prevNodes = []
        self.curNodes = []   
        self.addInputLayer()    
        self.addConvolutionalLayers()       
        self.addFlattenLayer()
        self.addLinearLayers()            
        self.addOutputLayer() 

        
    def addActivationLayer(self):
        for act in self.activationOptions:
            nodeName = 'L' + str(self.layer) + '_' + act.value
            self.addNode(nodeType=NodeType.ACTIVATION, name=nodeName, activationType=act)
        self.prevNodes = self.curNodes
        self.curNodes = []

          
    def addConvolutionalLayer(self):
        maxNumInputChannels = max(self.ALLOWED_NUMBER_OF_CONVOLUTION_CHANNELS)
        for kernel in self.ALLOWED_KERNEL_SIZES:
            for oc in self.ALLOWED_NUMBER_OF_CONVOLUTION_CHANNELS:
                nodeName = 'L' + str(self.layer) + '_' + str(kernel[0]) + 'x' + str(kernel[1]) + '_Conv(oc=' + str(oc) + ')'
                self.addNode(nodeType=NodeType.CONVOLUTION, 
                             name=nodeName, 
                             kernelSize=kernel, 
                             maxNumInputChannels=maxNumInputChannels, 
                             numOutputChannels=oc)
        self.prevNodes = self.curNodes
        self.curNodes = []

                         
    def addConvolutionalLayers(self):
        for i in range(self.numConvLayers):        
            self.layer += 1
            self.addConvolutionalLayer()                                        
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

    
    def addLinearLayer(self): 
        maxNumInputFeatures = max(self.ALLOWED_NUMBER_OF_LINEAR_FEATURES)
        for of in self.ALLOWED_NUMBER_OF_LINEAR_FEATURES:
            nodeName = 'L' + str(self.layer) + '_Linear(of=' + str(of) + ')' 
            self.addNode(nodeType=NodeType.LINEAR, 
                         name=nodeName, 
                         maxNumInFeatures=maxNumInputFeatures, 
                         numOutFeatures=of)
        self.prevNodes = self.curNodes
        self.curNodes = []

     
    def addLinearLayers(self):
        for i in range(self.numLinearLayers - 1):
            self.layer += 1            
            self.addLinearLayer()
            self.addActivationLayer()                                                  
        self.layer += 1
        nodeName = 'L' + str(self.layer) + '_Linear(of=' + str(10) + ')' 
        self.addNode(nodeType=NodeType.LINEAR, 
                     name=nodeName, 
                     maxNumInFeatures=max(self.ALLOWED_NUMBER_OF_LINEAR_FEATURES), 
                     numOutFeatures=self.numClasses)    
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


    def readGraph(self, filepath):
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"File not found: {filepath}")

        with open(filepath, 'rb') as file:
            self.graph = pickle.load(file)

              
    def render(self, dirPath=os.path.join(parentdir, "Graphs/GraphVisualizations/")):
        # Initialize the graph
        g = Digraph('G', filename = dirPath + 'enas_network_search_space')

        # Define graph attributes
        g.attr(rankdir='TB')  # 'TB' for top-to-bottom graph, 'LR' for left-to-right
        g.attr('node', shape='box', style='filled', color='lightgrey')
        g.attr(ranksep='2.0')  # Increase the space between layers, adjust the value as needed
        
        # Add Nodes
        for val in self.graph.values():
            node = val['node']
            g.node(node.name, node.displayName)
            
        # Add Edges
        for val in self.graph.values():
            node = val['node']
            for edge in val['edges']:
                g.edge(node.name, edge)
                
        # Specify the output format and render the graph
        g.format = 'png'
        filePath = dirPath + 'enas_network_search_space_visualization'
        g.render(filePath)

        return filePath + '.png'


    def writeGraph2File(self, filepath):
        # Assuming your_dict is the dictionary you want to save
        with open(filepath, 'wb') as file:
            pickle.dump(self.graph, file)

