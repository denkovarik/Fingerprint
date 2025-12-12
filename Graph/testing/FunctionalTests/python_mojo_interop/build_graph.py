import sys
from tqdm import tqdm
from Arch_Encoder import Arch_Encoder
import time
import os, io, sys, inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
grandparentdir = os.path.dirname(parentdir)
graphdir = os.path.dirname(grandparentdir)
sys.path.insert(0, graphdir)
from classes.Nodes import NodeType, NormalizationType, PoolingType, ActivationType
from classes.Nodes import Node, InputNode, OutputNode, NormalizationNode, ActivationNode, PoolingNode, FlattenNode, LinearNode, ConvolutionalNode
from classes.Graph import Graph



def read_and_split_file(file_path):
    all_dna = set()
    try:
        with open(file_path, 'r') as file:
            content = file.read()
            items = content.strip().split(',')
            for item in tqdm(items, desc=f"Reading file"):
                if item.strip() != "":
                    dna = item.strip()
                    all_dna.add(dna)
    except FileNotFoundError:
        print(f"Error: The file '{file_path}' was not found.")
        return set()
    except Exception as e:
        print(f"An error occurred: {e}")
        return set()
    return all_dna


if __name__ == "__main__":
    # Check command line args
    if len(sys.argv) < 2:
        print("Usage: python script.py <filename>")
        exit()
    
    arch_encoder = Arch_Encoder(input_shape=[4, 3, 32, 32], num_classes=10)
    
    file_path = sys.argv[1]
    all_dna = read_and_split_file(file_path)
    
    print(f"Total number of DNA extracted: {len(all_dna)}")
    
    min_len = 1000000
    max_len = 0
    graph_dna = set()
    cnt = 0
    for dna in all_dna:
        if len(dna) < min_len and len(dna) > 0:
            min_len = len(dna)
        elif len(dna) > max_len:
            max_len = len(dna)

    graph_networks = set()
    for dna in tqdm(all_dna, desc="Constructing Networks"):
        if len(dna) <= 60:
            graph_dna.add(dna)
            network = arch_encoder.translate(dna)
            if network != tuple():
                graph_networks.add(network)
            
    print(f"Number of DNA used in Graph: {len(graph_dna)}")
    print(f"Number of Networks in Graph: {len(graph_networks)}")
    
    graph = Graph()
    
    print("")
    for network in tqdm(graph_networks, desc="Compiling Graph"):
        try:
            graph.add_path_to_graph(network)
            #for node in network:
            #    print(node.name, end=" -> ")
            #print("")
            #graph.render()
            #user_input = input("Press Enter to continue: ")
        except:
            print(network)
            user_input = input("Press Enter to continue: ")
        
    graph.render()
        
    
    
    
        
    

