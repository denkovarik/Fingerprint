import numpy as np
import torch
import torch.nn as nn
import time
import os, io, sys, inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)
from classes.Nodes import NodeType, NormalizationType, PoolingType, ActivationType
from classes.Nodes import Node, InputNode, OutputNode, NormalizationNode, ActivationNode, PoolingNode, FlattenNode, LinearNode, ConvolutionalNode
import uuid


INPUT_CODON =  '0000000000000000'
INPUT_SHAPE = [4, 3, 32, 32]
OUTPUT_CODON = '1111111111111111'
NUM_CLASSES = 10

node_ids = {
    '1000010110011000': 'convolution',
    '1001111100011101': 'linear',
    '0011010110100001': 'normalize',
    '1110100110101010': 'activation',
    '0101101001101011': 'pooling'
}



def find_input_codon(dna):
    i = len(INPUT_CODON)
    start_ind = -1
    
    # Skip to input sequence
    while start_ind == -1 and i < len(dna):
        if dna[i-16:i] == INPUT_CODON:
            start_ind = i
        i += 1
     
    return i, start_ind
    
def find_output_codon(dna, i):
    stop_ind = -1
    
    # Skip to input sequence
    while stop_ind == -1 and i < len(dna):
        if dna[i-16:i] == OUTPUT_CODON:
            stop_ind = i
        i += 1
     
    return i, stop_ind

def read_kernel_size(ks):
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

def read_conv_node(dna, s, input_tensor):
    if s + 3 + 3 >= len(dna):
        return None, 1
    
    skip = 0
    in_channels = input_tensor.shape[1]
    out_chans_b = dna[s:s+3]
    out_channels_pow = max(2, int(out_chans_b, 2))
    out_channels = pow(2, out_channels_pow)
    skip += 3
    kernel_size = read_kernel_size(dna[s+skip:s+skip+3])
    skip += 3
    
    conv_node = ConvolutionalNode(name='convNode', kernel_size=kernel_size, 
                             maxNumInputChannels=128, 
                             maxNumOutputChannels=128, 
                             numOutputChannels=out_channels, layer=0,
                             pytorchLayerId=uuid.uuid4())
    
    m = nn.Conv2d(in_channels, out_channels, kernel_size)
    output_tensor = m(input_tensor)
    skip += 1
    
    return conv_node, output_tensor, skip

def read_linear_node(dna, s, input_tensor):
    if s >= len(dna):
        return None, input_tensor, 1
    
    skip = 0
    in_features = input_tensor.shape[1]
    out_features_b = dna[s:s+3]
    out_features_pow = max(1, int(out_features_b, 2))
    out_features = pow(2, out_features_pow)
    m = nn.Linear(in_features, out_features)
    output_tensor = m(input_tensor)
    skip += 3
    linear_node = LinearNode(name='linear', 
                      maxNumInFeatures=131072, 
                      maxNumOutFeatures=128,
                      numOutFeatures=out_features, 
                      layer=1, pytorchLayerId=uuid.uuid4())
    
    skip += 1
    
    return linear_node, output_tensor, skip
    
def read_pooling_node(dna, s, input_tensor):
    skip = 1
    kernelSize = 2
    stride = 2 
    pooling_node = PoolingNode(name='name', poolingType=PoolingType.MAX_POOLING)
    max_pool = nn.MaxPool2d(kernelSize, stride)
    output_tensor = max_pool(input_tensor)
    
    return pooling_node, output_tensor, skip

def translate_dna(dna: str, node_ids: dict) -> list:
    found_keys = []
    
    i, start_ind = find_input_codon(dna)
    i, stop_ind = find_output_codon(dna, i)
        
    if start_ind == -1 or stop_ind == -1:
        return tuple()
    
    input_tensor = torch.rand(INPUT_SHAPE)
    input_node = InputNode(inputShape=input_tensor.shape)
    in_features = 1
    
    found_keys.append(input_node)
    flatener = nn.Flatten()
    
    i = start_ind + 16
    
    while i < stop_ind:
        skip = 1
        if dna[i-16:i] in node_ids.keys():
            id = dna[i-16:i]
            node = node_ids[id]        
            if node == 'convolution' and len(input_tensor.shape) == 4:
                node, out_tensor, skip = read_conv_node(dna, i, input_tensor)
                found_keys.append(node)
                input_tensor = out_tensor
            elif node == 'linear':
                if len(input_tensor.shape) > 2:
                    input_tensor = flatener(input_tensor)
                    found_keys.append(FlattenNode(name='flatten'))
                node, out_tensor, skip = read_linear_node(dna, i, input_tensor)
                found_keys.append(node)
                input_tensor = out_tensor
            elif node == 'normalize':
                normNode = NormalizationNode('normNode', NormalizationType.BATCH_NORM, 
                    input_tensor.shape, pytorchLayerId=uuid.uuid4())
                found_keys.append(normNode)
            elif node == 'activation':
                found_keys.append(ActivationNode(name='relu_activation', activationType=ActivationType.RELU))
            elif node == 'pooling' and len(input_tensor.shape) == 4:
                node, out_tensor, skip = read_pooling_node(dna, i, input_tensor)
                found_keys.append(node)
                input_tensor = out_tensor
                
        i += skip
    
    if len(input_tensor.shape) > 2:
        flatener = nn.Flatten()
        input_tensor = flatener(input_tensor)
        found_keys.append(FlattenNode(name='flatten'))
    
    in_features = input_tensor.shape[1]    
    linear_node = LinearNode(name='linear_out', 
                      maxNumInFeatures=128, 
                      maxNumOutFeatures=128,
                      numOutFeatures=NUM_CLASSES, 
                      layer=1, pytorchLayerId=uuid.uuid4())    
    found_keys.append(linear_node)
    found_keys.append(OutputNode())
    
    # Return only the keys in order as a tuple
    return tuple(key for key in found_keys)




print("Node Type       Node ID")
print("--------------------------------")
for id, node_type in node_ids.items():
    print(f"{node_type}: \t{id}")

print("")


rnas = set()

for i in range(1000):
    dna_array = np.random.randint(0, 2, size=2048)
    dna = INPUT_CODON + ''.join(dna_array.astype(str)) + OUTPUT_CODON        
    rnas.add(translate_dna(dna, node_ids))

print("\n")
print("Gene Pool")
print("-----------------------------------")
for rna in rnas:
    print(rna)