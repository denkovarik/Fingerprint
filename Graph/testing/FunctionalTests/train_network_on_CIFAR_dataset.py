import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import numpy as np
import random
import os, io, sys, inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
graphdir = os.path.dirname(parentdir)
sys.path.insert(0, graphdir)
from classes.ENAS import ENAS
from utils import renderGraph


def load_cifar10(data_dir='Datasets/CIFAR-10'):
    """
    Checks if the CIFAR-10 dataset exists in the specified directory.
    If not, downloads the dataset.
    
    Args:
        Data_dir (str): Directory to check for the dataset and download to.
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
    
    print("CIFAR-10 training and test sets are loaded and available in ", data_dir)
    
    return trainset, testset
    
    
def test_network(model, testset, device, batch_size=32):
    """
    Tests the neural network model on the provided test dataset.
    
    Args:
        Model (CustomCNN): The neural network model
        testset (torchvision.datasets): The test dataset.
        batch_size (int): Batch size for testing
        
    Returns:
        float: The accuracy of the model on the test set
    """
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False)
    correct = 0
    total = 0
    model.to(device)
    model.eval()
    
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
    accuracy = 100 * correct / total
    return accuracy
    
    
def train_network(model, trainset, device, batch_size=32, num_epochs=10):
    """
    Trains a neural network on a given dataset.

    Args:
        model (CustomCNN): The neural network model to train.
        trainloader (torch.utils.data.DataLoader): DataLoader for the training dataset.
        device (str): Device to train the model on ('cuda' or 'cpu').
        num_epochs (int): Number of epochs to train the model.

    Returns:
        torch.nn.Module: The trained model.

    This function trains the model on the provided dataset for the specified number of epochs.
    It uses CrossEntropyLoss and the Adam optimizer for training. After training, it prints the loss for each epoch and returns the trained model.
    """
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)
    
    model.to(device)
    model.train()
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    for epoch in range(num_epochs):
        running_loss = 0.0
        for images, labels in trainloader:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
        print(f'Epoch {epoch + 1}, Loss: {running_loss / len(trainloader)}')
        
    print('Finished Training')
    return model
 
    
trainset, testset = load_cifar10()

enas = ENAS(inputShape=torch.Size([4, 3, 32, 32]))
enas.construct()

sample = [1, 0, 1, 1, 0, 5, 1, 1, 1, 0, 3, 1, 1, 1, 0, 0, 0]
enas.sampleArchitecture(sample)

# Render the graph
renderGraph(enas.graph, os.path.join(currentdir, 'Temp'))

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

enas.sample.to(device)

untrained_accuracy = test_network(enas.sample, testset, device)
print(f'Accuracy of the untrained network on the test images: {untrained_accuracy}%')

train_network(enas.sample, trainset, device, batch_size=256, num_epochs=25)

trained_accuracy = test_network(enas.sample, testset, device)
print(f'Accuracy of the trained network on the test images: {trained_accuracy}%')
    
if trained_accuracy > 70:
    print("\U0001F60E")
