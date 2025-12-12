import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import numpy as np
import random
import time
import uuid
import os, io, sys, inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
grandparentdir = os.path.dirname(parentdir)
graphdir = os.path.dirname(grandparentdir)
sys.path.insert(0, graphdir)
from classes.ENAS import ENAS, CustomCNN
from classes.Graph import Graph
from utils import renderGraph
from arch_code_reader import Arch_Encoder, Architecture, mutate, reproduce
from classes.Nodes import NodeType, NormalizationType, PoolingType, ActivationType
from classes.Nodes import Node, InputNode, OutputNode, NormalizationNode, ActivationNode, PoolingNode, FlattenNode, LinearNode, ConvolutionalNode
from utils import renderGraph
from concurrent.futures import ThreadPoolExecutor


def load_cifar10(data_dir='Datasets/CIFAR-10', batch_size=32):
    """
    Loads the CIFAR-10 dataset and precomputes batches of the specified size.
    
    Args:
        data_dir (str): Directory to check for the dataset and download to.
        batch_size (int): Size of each batch to precompute.
    
    Returns:
        Tuple[List[torch.Tensor], List[torch.Tensor], List[torch.Tensor], List[torch.Tensor]]:
            Batches of training images, training labels, test images, and test labels.
    """
    # Define transformations
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # Load or download the trainset
    trainset = torchvision.datasets.CIFAR10(root=data_dir, train=True, download=True, transform=transform)
    
    # Load or download the testset
    testset = torchvision.datasets.CIFAR10(root=data_dir, train=False, download=True, transform=transform)

    print(f"CIFAR-10 training and test sets are loaded and available in {data_dir}")

    # Precompute training batches
    train_batches = []
    train_labels = []
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)
    for images, labels in trainloader:
        train_batches.append(images)
        train_labels.append(labels)

    # Precompute test batches
    test_batches = []
    test_labels = []
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False)
    for images, labels in testloader:
        test_batches.append(images)
        test_labels.append(labels)

    return train_batches, train_labels, test_batches, test_labels
    
    
def test_network(model, test_images, test_labels, device, batch_size=32):
    correct = 0
    total = 0
    model.to(device)
    model.eval()
    
    num_batches = len(test_images)
    
    with torch.no_grad():
        for i in range(num_batches):
            images = train_images[i].to(device)
            labels = train_labels[i].to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
    accuracy = 100 * correct / total
    return accuracy
    
    
class AccuracyTracker:
    def __init__(self, max_epochs):
        self.high_val_acc = 0.0
        self.high_val_acc_cnt = 0
        self.epoch = 0
        self.max_epochs = max_epochs
        self.improving = True
    
    def is_improving(self, accuracy):
        if accuracy > self.high_val_acc:
            self.high_val_acc = accuracy
            self.high_val_acc_cnt = 0
        else:
            self.high_val_acc_cnt += 1
        
        self.epoch += 1
        
        if self.high_val_acc_cnt > 10 or self.epoch >= self.max_epochs:
            self.improving = False
        
        if not self.improving and accuracy < self.high_val_acc - 1:
            return True
        return self.improving
    
    
def train_network(model, train_images, train_labels, test_images, test_labels, device, max_epochs=10):    
    model.to(device)
    model.train()
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    num_batches = len(train_images)
    epoch = 1
    
    tracker = AccuracyTracker(max_epochs)
    validation_accuracy = 0.0
    
    while epoch <= max_epochs:
        running_loss = 0.0
        for i in range(num_batches):
            images = train_images[i].to(device)
            labels = train_labels[i].to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
        validation_accuracy = test_network(model, test_images, test_labels, device)
        #print(f'Epoch {epoch + 1}: Loss: {running_loss / num_batches:.4f}   Validation Accuracy: {validation_accuracy:.2f}%')

        epoch += 1
        
    #print('Finished Training')
    return model
    
    
def construct_model(phenotype):
    layers = []
    out = torch.rand(input_shape)
    for i in range(len(phenotype.architecture)):
        layer = phenotype.architecture[i].getLayer(out.shape)
        if layer is not None:
            layers.append(layer)
            out = layers[-1](out)
    model = CustomCNN(layers, input_shape)
    return model
    
    
def train_population(population, device):
    cpu = torch.device('cpu') 
    population_ranked = sorted(population, key=lambda x: x.top_score, reverse=True)
    to_train = []
    for i in range(len(population_ranked)):
        if i < 10:
            to_train.append(i)
        elif population_ranked[i].top_score < 0.01:
            to_train.append(i)
        elif random.random() < 0.3:
            to_train.append(i)
    
    for i in to_train:
        try:
            model = construct_model(population_ranked[i])
            train_network(model, train_images, train_labels, test_images, test_labels, device, max_epochs=3)
            accuracy = test_network(model, test_images, test_labels, device)
            if population_ranked[i].top_score is None or accuracy > population_ranked[i].top_score:
                population_ranked[i].top_score = accuracy
            model.to(cpu)
            print(population_ranked[i])
        except:
            print(f"Error occurred with phonotype: {phenotype}")
    print("")
    return set(population_ranked)
 
 
def get_offspring(population):
    num_chosen = len(population) * 0.8
    ranked_list = sorted(population, key=lambda x: x.top_score, reverse=True)
    chosen = set()
    ind = 0
    while len(chosen) < num_chosen:
        if ranked_list[ind].architecture != tuple():
            prop_of_selection = float(len(ranked_list) - ind) / len(ranked_list)
            if random.random() < prop_of_selection:
                chosen.add(ranked_list[ind])
        ind += 1
        if ind >= len(ranked_list):
            ind = 0
    offspring = reproduce(chosen, population)              
    return set(offspring.values())


if __name__ == "__main__":    
    train_images, train_labels, test_images, test_labels = load_cifar10(data_dir=os.path.join(currentdir, 'Datasets/CIFAR-10'), batch_size=2048)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")    
    
    input_tensor = torch.rand([4, 3, 32, 32])
    input_shape = input_tensor.shape
    arch_encoder = Arch_Encoder(input_shape=[4, 3, 32, 32], num_classes=10)
    
    population = set()
    dna = ''

    # Seeding the population
    dna_array = np.random.randint(0, 8, size=1)
    dna = ''.join(dna_array.astype(str)) 
    dna = dna + arch_encoder.OUTPUT_CODON + arch_encoder.node_ids['linear'] + arch_encoder.node_ids['activation'] + '11111111'

    phenotype = Architecture(arch_encoder.translate(dna), {dna})
    population.add(phenotype)
    offspring = set()

    num_generations = 100
    generation_num = 1

    with ThreadPoolExecutor(max_workers=2) as executor:
        while generation_num <= num_generations: 
            print("\n")
            print(f"Generation {generation_num}")
            print("-----------------------------------")
            
            # Submit the reproduction task
            reproduce_future = executor.submit(get_offspring, population)
            
            print("Training Population")
            # Submit the training task
            train_future = executor.submit(train_population, population, device)
            
            # Wait for both tasks to finish
            offspring = reproduce_future.result()
            population = train_future.result()        
                    
            for phenotype in population & offspring:
                for child in offspring:
                    if child == phenotype:
                        phenotype.genotypes.update(child.genotypes)
                        break
                        
            # Add new phenotypes to population
            population.update(offspring)

            generation_num += 1
            
            sorted_list = sorted(population, key=lambda x: x.top_score, reverse=True)
        
            for arch in sorted_list:
                print(arch)
