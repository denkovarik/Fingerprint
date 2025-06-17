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
graphdir = os.path.dirname(parentdir)
sys.path.insert(0, graphdir)
from classes.ENAS import ENAS, CustomCNN
from classes.Graph import Graph
from utils import renderGraph
from arch_code_reader import Arch_Encoder, Architecture, mutate
from classes.Nodes import NodeType, NormalizationType, PoolingType, ActivationType
from classes.Nodes import Node, InputNode, OutputNode, NormalizationNode, ActivationNode, PoolingNode, FlattenNode, LinearNode, ConvolutionalNode
from utils import renderGraph



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
        
        if self.high_val_acc_cnt > 100 or self.epoch >= self.max_epochs:
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
    epoch = 0
    
    tracker = AccuracyTracker(max_epochs)
    validation_accuracy = 0.0
    
    while tracker.is_improving(validation_accuracy):
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
        print(f'Epoch {epoch + 1}: Loss: {running_loss / num_batches:.4f}   Validation Accuracy: {validation_accuracy:.2f}%')

        epoch += 1
        
    print('Finished Training')
    return model
 
 
if __name__ == "__main__":    
    input_tensor = torch.rand([4, 3, 32, 32])
    input_shape = input_tensor.shape
    input_node = InputNode(inputShape=input_tensor.shape)
    normNode = NormalizationNode('norm1', NormalizationType.BATCH_NORM, 
        input_tensor.shape, pytorchLayerId=uuid.uuid4())
    conv_node = ConvolutionalNode(name='conv1', kernel_size=3, 
                         maxNumInputChannels=128, 
                         maxNumOutputChannels=128, 
                         numOutputChannels=16, layer=0,
                         pytorchLayerId=uuid.uuid4())
    conv_node.setSharedLayer(conv_node.constructLayer())
    output_tensor = conv_node(input_tensor)
    relu_node = ActivationNode(name='relu_activation', activationType=ActivationType.RELU)
    flatener = nn.Flatten()
    output_tensor = flatener(output_tensor)
    flat_node = FlattenNode(name='flatten')
    linear_node = LinearNode(name='linear', 
                  maxNumInFeatures=14400, 
                  maxNumOutFeatures=10,
                  numOutFeatures=10, 
                  layer=1, pytorchLayerId=uuid.uuid4())
    linear_node.setSharedLayer(linear_node.constructLayer())
    output_tensor = linear_node(output_tensor)
    relu_node_out = ActivationNode(name='relu_activation_out', activationType=ActivationType.RELU)
    
    nodes = [normNode, conv_node, relu_node, flat_node, linear_node, relu_node_out]
    layers = []
    
    out = torch.rand(input_shape)
    for i in range(len(nodes)-1):
        layer = nodes[i].getLayer(out.shape)
        if layer is not None:
            layers.append(layer)
            out = layers[-1](out)
    
    model = CustomCNN(layers, input_shape)
    
    print(model)
    
    # Render the graph
    #renderGraph(enas.graphHandler, os.path.join(currentdir, 'Temp'))
    
    train_images, train_labels, test_images, test_labels = load_cifar10(data_dir=os.path.join(currentdir, 'Datasets/CIFAR-10'), batch_size=2048)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    untrained_accuracy = test_network(model, test_images, test_labels, device)
    print(f'Accuracy of the untrained network on the test images: {untrained_accuracy}%')
    
    start_time = time.time()
    train_network(model, train_images, train_labels, test_images, test_labels, device, max_epochs=1000)
    end_time = time.time()
    elapsed_time = end_time - start_time

    trained_accuracy = test_network(model, test_images, test_labels, device)
    print(f'Accuracy of the trained network on the test images: {trained_accuracy}%')
    print(f'Training time: {elapsed_time}')
    
    if trained_accuracy > 90:
        print("\U0001F60E")