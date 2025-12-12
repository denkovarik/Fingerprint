import numpy as np
import torch
import torch.nn as nn
import time
import os, io, sys, inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
grandparentdir = os.path.dirname(parentdir)
graphdir = os.path.dirname(grandparentdir)
sys.path.insert(0, graphdir)
from classes.Nodes import NodeType, NormalizationType, PoolingType, ActivationType
from classes.Nodes import Node, InputNode, OutputNode, NormalizationNode, ActivationNode, PoolingNode, FlattenNode, LinearNode, ConvolutionalNode
import uuid
import random
import ast


class Arch_Encoder:
    def __init__(self, input_shape=[4, 3, 32, 32], num_classes=10):
        self.INPUT_CODON =  '00000000'
        self.INPUT_SHAPE = input_shape
        self.OUTPUT_CODON = '11111111'
        self.NUM_CLASSES = num_classes

        self.node_types = {
            '10000101': 'convolution',
            '10011111': 'linear',
            '00110101': 'normalize',
            '11101001': 'activation',
            '01011010': 'pooling'
        }
        self.node_ids = {
            'convolution':  '10000101',
            'linear':       '10011111',
            'normalize':    '00110101',
            'activation':   '11101001',
            'pooling':      '01011010'
        }

    def find_input_codon(self, dna):
        i = len(self.INPUT_CODON)
        start_ind = -1
        
        # Skip to input sequence
        while start_ind == -1 and i < len(dna):
            if dna[i-8:i] == self.INPUT_CODON:
                start_ind = i
            i += 1
         
        return i, start_ind
        
    def find_output_codon(self, dna, i):
        stop_ind = -1
        
        while stop_ind == -1 and i < len(dna):
            if dna[i-8:i] == self.OUTPUT_CODON:
                stop_ind = i
            i += 1
         
        return i, stop_ind

    def read_kernel_size(self, ks):
        translated_ks = 3
        if ks == '000':
            translated_ks = 3
        elif ks == '001':
            translated_ks = 5
        elif ks == '010':
            translated_ks = 7
        elif ks == '011':
            translated_ks = 3
        elif ks == '100':
            translated_ks = 5
        elif ks == '101':
            translated_ks = 7
        elif ks == '110':
            translated_ks = 3
        elif ks == '111':
            translated_ks = 5
        return translated_ks

    def read_conv_node(self, dna, s, input_tensor):
        if s + 3 + 3 >= len(dna):
            return None, input_tensor, 1
        
        skip = 0
        in_channels = input_tensor.shape[1]
        out_chans_b = dna[s:s+3]
        out_channels_pow = int(out_chans_b, 2)
        if out_channels_pow < 2:
            out_channels_pow = 2
        out_channels = pow(2, out_channels_pow)
        skip += 3
        kernel_size = self.read_kernel_size(dna[s+skip:s+skip+3])
        skip += 3
        
        conv_node = ConvolutionalNode(name=f'convNode_{self.conv_cnt}', kernel_size=kernel_size, 
                                 maxNumInputChannels=in_channels, 
                                 maxNumOutputChannels=out_channels, 
                                 numOutputChannels=out_channels, layer=self.conv_cnt,
                                 pytorchLayerId=uuid.uuid4())
        conv_node.name = conv_node.name + '->' + conv_node.displayName
        try:
            m = nn.Conv2d(in_channels, out_channels, kernel_size)
            output_tensor = m(input_tensor)
        except:
            return None, input_tensor, 1
        
        skip += 1
        self.conv_cnt += 1
        return conv_node, output_tensor, skip

    def read_linear_node(self, dna, s, input_tensor, output=False):
        skip = 0
        in_features = input_tensor.shape[1]
        
        if output:
            out_features = self.NUM_CLASSES
            m = nn.Linear(in_features, out_features)
            output_tensor = m(input_tensor)
            skip += 1
            linear_node = LinearNode(name=f'linear_out_{self.layer_out}', 
                              maxNumInFeatures=131072, 
                              maxNumOutFeatures=128,
                              numOutFeatures=out_features, 
                              layer=-1, pytorchLayerId=uuid.uuid4())
            linear_node.name = linear_node.name + '->' + linear_node.displayName
            return linear_node, output_tensor, skip
        
        if s + 3 >= len(dna):
            return None, input_tensor, 1
        
        skip = 0
        in_features = input_tensor.shape[1]
        out_features_b = dna[s:s+3]
        out_features_pow = int(out_features_b, 2)
        if out_features_pow == 0:
            out_features_pow = 1
        out_features = pow(2, out_features_pow)
        m = nn.Linear(in_features, out_features)
        output_tensor = m(input_tensor)
        skip += 3
        linear_node = LinearNode(name=f'linear_{self.linear_cnt}', 
                          maxNumInFeatures=in_features, 
                          maxNumOutFeatures=in_features,
                          numOutFeatures=out_features, 
                          layer=self.linear_cnt, pytorchLayerId=uuid.uuid4())
        linear_node.name = linear_node.name + '->' + linear_node.displayName
        self.linear_cnt += 1
        skip += 1
        return linear_node, output_tensor, skip
        
    def read_pooling_node(self, dna, s, input_tensor):
        skip = 1
        kernelSize = 2
        stride = 2 
        try:
            pooling_node = PoolingNode(name=f'max_pooling_{len(self.found_keys)+1}', poolingType=PoolingType.MAX_POOLING)
            max_pool = nn.MaxPool2d(kernelSize, stride)
            output_tensor = max_pool(input_tensor)
        except:
            return None, input_tensor, 1
        self.pooling_cnt += 1
        
        return pooling_node, output_tensor, skip

    def translate(self, dna: str) -> list:
        self.found_keys = []
        
        start_ind = 0
        i, stop_ind = self.find_output_codon(dna, 0)
            
        if start_ind == -1 or stop_ind == -1:
            return tuple()
        
        input_tensor = torch.rand(self.INPUT_SHAPE)
        input_node = InputNode(inputShape=input_tensor.shape)
        in_features = 1
        
        self.found_keys.append(input_node)
        flatener = nn.Flatten()
        
        i = start_ind + 8
        self.conv_cnt = 0
        self.linear_cnt = 0
        self.act_cnt = 0
        self.pooling_cnt = 0
        self.batch_norm_cnt = 0
        
        while i < stop_ind:
            skip = 1
            if dna[i-8:i] in self.node_types.keys():
                id = dna[i-8:i]
                node = self.node_types[id]        
                if node == 'convolution' and len(input_tensor.shape) == 4:
                    node, input_tensor, skip = self.read_conv_node(dna, i, input_tensor)
                    if node is not None:
                        self.found_keys.append(node)
                elif node == 'linear':
                    if len(input_tensor.shape) > 2:
                        input_tensor = flatener(input_tensor)
                        self.found_keys.append(FlattenNode(name='flatten'))
                    node, input_tensor, skip = self.read_linear_node(dna, i, input_tensor)
                    if node is not None:
                        self.found_keys.append(node)
                elif node == 'normalize' and len(input_tensor.shape) == 4:
                    if isinstance(self.found_keys[-1], NormalizationNode):
                        return tuple()
                    normNode = NormalizationNode(f'normNode_{{len(self.found_keys)+1}}', NormalizationType.BATCH_NORM, 
                        input_tensor.shape, pytorchLayerId=uuid.uuid4())
                    if node is not None:
                        self.found_keys.append(normNode)
                        self.batch_norm_cnt += 1
                elif node == 'activation':
                    if isinstance(self.found_keys[-1], ActivationNode):
                        return tuple()
                    if len(input_tensor.shape) == 4:
                        self.found_keys.append(ActivationNode(name=f'relu_activation_{str(len(self.found_keys))}', activationType=ActivationType.RELU))
                        self.act_cnt += 1
                    else:
                        self.found_keys.append(ActivationNode(name=f'relu_activation_flat_{str(len(self.found_keys))}', activationType=ActivationType.RELU))
                        self.act_cnt += 1
                elif node == 'pooling' and len(input_tensor.shape) == 4 and not (len(self.found_keys) == 0 or isinstance(self.found_keys[-1], PoolingNode)):
                    node, input_tensor, skip = self.read_pooling_node(dna, i, input_tensor)
                    if node is not None:
                        self.found_keys.append(node)
                    
            i += skip
         
        if len(input_tensor.shape) > 2:
            input_tensor = flatener(input_tensor)
            self.found_keys.append(FlattenNode(name='flatten')) 
            
        self.layer_out = 0
         
        valid_arch = False  
        out_linear_cnt = 0
        while i < len(dna):
            skip = 1
            if dna[i-8:i] in self.node_types.keys():
                id = dna[i-8:i]
                node = self.node_types[id]        
                if node == 'linear' and out_linear_cnt == 0:
                    node, input_tensor, skip = self.read_linear_node(dna, i, input_tensor, output=True)
                    if node is not None:
                        self.found_keys.append(node)
                        valid_arch = True
                        out_linear_cnt += 1
                        self.layer_out += 1
                elif node == 'activation':
                    if isinstance(self.found_keys[-1], ActivationNode):
                        return tuple()
                    if node is not None:
                        self.found_keys.append(ActivationNode(name=f'relu_activation_out_{str(self.layer_out)}', activationType=ActivationType.RELU))
                        self.layer_out += 1
                    
            i += skip
        
        self.found_keys.append(OutputNode())
        
        if valid_arch:
            return tuple(key for key in self.found_keys)
            
        return tuple()
