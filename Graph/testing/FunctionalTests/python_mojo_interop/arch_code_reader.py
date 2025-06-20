import numpy as np
import torch
import torch.nn as nn
import time
import os, io, sys, inspect
import max.mojo.importer
sys.path.insert(0, "")
import arch_coder_m
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
        
        conv_node = ConvolutionalNode(name='convNode', kernel_size=kernel_size, 
                                 maxNumInputChannels=in_channels, 
                                 maxNumOutputChannels=out_channels, 
                                 numOutputChannels=out_channels, layer=0,
                                 pytorchLayerId=uuid.uuid4())
        
        try:
            m = nn.Conv2d(in_channels, out_channels, kernel_size)
            output_tensor = m(input_tensor)
        except:
            return None, input_tensor, 1
        
        skip += 1
        conv_node.setSharedLayer(conv_node.constructLayer())
        
        return conv_node, output_tensor, skip

    def read_linear_node(self, dna, s, input_tensor, output=False):
        skip = 0
        in_features = input_tensor.shape[1]
        
        if output:
            out_features = self.NUM_CLASSES
            m = nn.Linear(in_features, out_features)
            output_tensor = m(input_tensor)
            skip += 1
            linear_node = LinearNode(name='linear', 
                              maxNumInFeatures=131072, 
                              maxNumOutFeatures=128,
                              numOutFeatures=out_features, 
                              layer=1, pytorchLayerId=uuid.uuid4())
            linear_node.setSharedLayer(linear_node.constructLayer())
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
        linear_node = LinearNode(name='linear', 
                          maxNumInFeatures=in_features, 
                          maxNumOutFeatures=in_features,
                          numOutFeatures=out_features, 
                          layer=1, pytorchLayerId=uuid.uuid4())
        
        skip += 1
        linear_node.setSharedLayer(linear_node.constructLayer())
        return linear_node, output_tensor, skip
        
    def read_pooling_node(self, dna, s, input_tensor):
        skip = 1
        kernelSize = 2
        stride = 2 
        pooling_node = PoolingNode(name='name', poolingType=PoolingType.MAX_POOLING)
        max_pool = nn.MaxPool2d(kernelSize, stride)
        output_tensor = max_pool(input_tensor)
        
        return pooling_node, output_tensor, skip

    def translate(self, dna: str) -> list:
        found_keys = []
        
        start_ind = 0
        i, stop_ind = self.find_output_codon(dna, 0)
            
        if start_ind == -1 or stop_ind == -1:
            return tuple()
        
        input_tensor = torch.rand(self.INPUT_SHAPE)
        input_node = InputNode(inputShape=input_tensor.shape)
        in_features = 1
        
        found_keys.append(input_node)
        flatener = nn.Flatten()
        
        i = start_ind + 8
        
        while i < stop_ind:
            skip = 1
            if dna[i-8:i] in self.node_types.keys():
                id = dna[i-8:i]
                node = self.node_types[id]        
                if node == 'convolution' and len(input_tensor.shape) == 4:
                    node, out_tensor, skip = self.read_conv_node(dna, i, input_tensor)
                    found_keys.append(node)
                    input_tensor = out_tensor
                elif node == 'linear':
                    if len(input_tensor.shape) > 2:
                        input_tensor = flatener(input_tensor)
                        found_keys.append(FlattenNode(name='flatten'))
                    node, out_tensor, skip = self.read_linear_node(dna, i, input_tensor)
                    found_keys.append(node)
                    input_tensor = out_tensor
                elif node == 'normalize' and len(input_tensor.shape) == 4:
                    normNode = NormalizationNode('normNode', NormalizationType.BATCH_NORM, 
                        input_tensor.shape, pytorchLayerId=uuid.uuid4())
                    found_keys.append(normNode)
                elif node == 'activation':
                    found_keys.append(ActivationNode(name='relu_activation', activationType=ActivationType.RELU))
                elif node == 'pooling' and len(input_tensor.shape) == 4:
                    node, out_tensor, skip = self.read_pooling_node(dna, i, input_tensor)
                    found_keys.append(node)
                    input_tensor = out_tensor
                    
            i += skip
         
        valid_arch = False  
        while i < len(dna):
            skip = 1
            if dna[i-8:i] in self.node_types.keys():
                id = dna[i-8:i]
                node = self.node_types[id]        
                if node == 'linear':
                    if len(input_tensor.shape) > 2:
                        input_tensor = flatener(input_tensor)
                        found_keys.append(FlattenNode(name='flatten'))
                    node, out_tensor, skip = self.read_linear_node(dna, i, input_tensor, output=True)
                    found_keys.append(node)
                    input_tensor = out_tensor
                    valid_arch = True
                elif node == 'normalize' and len(input_tensor.shape) == 4:
                    normNode = NormalizationNode('normNode', NormalizationType.BATCH_NORM, 
                        input_tensor.shape, pytorchLayerId=uuid.uuid4())
                    found_keys.append(normNode)
                elif node == 'activation':
                    found_keys.append(ActivationNode(name='relu_activation', activationType=ActivationType.RELU))
                    
            i += skip
        
        found_keys.append(OutputNode())
        
        if valid_arch:
            return tuple(key for key in found_keys)
            
        return tuple()


def mutate(dna: str, mutation_rate: float) -> str:
    # Convert the string to a NumPy array of integers
    dna_array = np.fromiter(dna, dtype=np.int8)  
    # Create a mask for mutations
    mask = np.random.random(dna_array.shape) < mutation_rate   
    # Flip bits where the mask is True
    dna_array[mask] ^= 1  # XOR with 1 flips the bits   
    # Convert back to string
    return ''.join(map(str, dna_array))
    
    
class Architecture:
    def __init__(self, architecture, genotypes):
        self.architecture = architecture
        self.genotypes = set()
        self.genotypes.update(genotypes)
        self.top_score = 0.0

    def __str__(self):
        return f"Accuracy: {self.top_score:.2f}%    Architecture: {self.architecture}"

    def __repr__(self):
        return self.__str__()

    def __eq__(self, other):
        if isinstance(other, Architecture):
            return (self.architecture == other.architecture)
        return False

    def __hash__(self):
        architecture_hash = tuple(hash(node) for node in self.architecture)
        return hash((architecture_hash))


def reproduce(chosen, population):
    chosen_genotypes = set()
    pop_genotypes = set()
    for phenotype in chosen:
        for dna in phenotype.genotypes:
            chosen_genotypes.add(dna)
            
    for phenotype in population:
        for dna in phenotype.genotypes:
            pop_genotypes.add(dna)    
    
    chosen_dna_str = ""
    pop_dna_str = ""

    for dna in pop_genotypes:
        pop_dna_str = pop_dna_str + dna + ","
    with open(pop_filename, 'w') as file:
        file.write(pop_dna_str)
        
    for dna in chosen_genotypes:
        chosen_dna_str = chosen_dna_str + dna + ","
    with open(chosen_filename, 'w') as file:
        file.write(chosen_dna_str)

    arch_coder_m.reproduce(chosen_filename, pop_filename, output_filename, 10, 10)
    
    offspring_temp = {}
    with open(output_filename, 'r') as file:
        offspring_str = file.read()
        offspring_temp = ast.literal_eval(offspring_str)
       
    offspring = {} 
    for phenotype_key in offspring_temp.keys():
        offspring[phenotype_key] = Architecture(phenotype_key, offspring_temp[phenotype_key])
        
    return offspring

if __name__ == "__main__":
    arch_encoder = Arch_Encoder(input_shape=[4, 3, 32, 32], num_classes=10)

    print("Node Type       Node ID")
    print("--------------------------------")
    for id, node_type in arch_encoder.node_types.items():
        print(f"{node_type}: \t{id}")

    print("")


    population = set()
    dna = ''

    # Seeding the population
    dna_array = np.random.randint(0, 2, size=8)
    dna = ''.join(dna_array.astype(str)) 
    dna = dna + arch_encoder.OUTPUT_CODON + arch_encoder.node_ids['linear'] + arch_encoder.node_ids['activation'] + '11111111'

    phenotype = Architecture(arch_encoder.translate(dna), {dna})
    population.add(phenotype)

    print("\n")
    print("Generation 1")
    print("-----------------------------------")
    for phenotype in population:
        print(phenotype)

    num_generations = 10
    generation_num = 2
    num_offspring = 10
    pop_filename = 'pop_dna.txt'
    chosen_filename = 'chosen_dna.txt'
    output_filename = 'offspring_dna.txt'

    while generation_num <= num_generations: 
        offspring = reproduce(population, population)
                
        offspring_set = set(offspring.values())
        for phenotype in population & offspring_set:
            if phenotype in offspring.keys():
                phenotype.genotypes.update(offspring[phenotype].genotypes)
    
        # Add new phenotypes to population
        population.update(offspring_set)
            
        print(f"Generation {generation_num}\n\tNumber of phenotypes = {len(population)}")
        #print("\n\n")
        #print(f"Generation {generation_num}")
        #print("-----------------------------------")
        #for child in population:
        #    if len(child.architecture) > 0:
        #        print(child)
    
        time.sleep(1)
        generation_num += 1
