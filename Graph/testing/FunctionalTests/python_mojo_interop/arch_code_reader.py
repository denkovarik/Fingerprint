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

from Arch_Encoder import Arch_Encoder


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
    pop_filename = 'pop_dna.txt'
    chosen_filename = 'chosen_dna.txt'
    output_filename = 'offspring_dna.txt'
    arch_encoder = Arch_Encoder(input_shape=[4, 3, 32, 32], num_classes=10)
    
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
        dna = next(iter(offspring_temp[phenotype_key]))
        offspring[phenotype_key] = Architecture(arch_encoder.translate(dna), offspring_temp[phenotype_key])
        
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
    dna_array = np.random.randint(0, 2, size=32)
    dna = ''.join(dna_array.astype(str)) 
    dna = dna + arch_encoder.OUTPUT_CODON + arch_encoder.node_ids['linear'] + arch_encoder.node_ids['activation'] + '11111111'

    phenotype = Architecture(arch_encoder.translate(dna), {dna})
    population.add(phenotype)

    print("\n")
    print("Generation 1")
    print("-----------------------------------")
    for phenotype in population:
        print(phenotype)

    num_generations = 100
    generation_num = 2
    num_offspring = 10

    while generation_num <= num_generations: 
        offspring = reproduce(population, population)
                
        offspring_set = set(offspring.values())
        for phenotype in population & offspring_set:
            if phenotype in offspring.keys():
                phenotype.genotypes.update(offspring[phenotype].genotypes)
    
        # Add new phenotypes to population
        population.update(offspring_set)
            
        #print(f"Generation {generation_num}\n\tNumber of phenotypes = {len(population)}")
        print("\n\n")
        print(f"Generation {generation_num}")
        print("-----------------------------------")
        for child in population:
            if len(child.architecture) > 0:
                print(child)
    
        generation_num += 1
