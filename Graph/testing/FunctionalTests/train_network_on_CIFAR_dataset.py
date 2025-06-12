import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import numpy as np
import random
import time
import os, io, sys, inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
graphdir = os.path.dirname(parentdir)
sys.path.insert(0, graphdir)
from classes.ENAS import ENAS
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
    
    
def train_network(model, train_images, train_labels, test_images, test_labels, device, num_epochs=10):    
    model.to(device)
    model.train()
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    num_batches = len(train_images)
    
    for epoch in range(num_epochs):
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
            
        validation_accuracy = test_network(enas.sample, test_images, test_labels, device)
        print(f'Epoch {epoch + 1}: Loss: {running_loss / num_batches:.4f}   Validation Accuracy: {validation_accuracy:.2f}%')
        
    print('Finished Training')
    return model
 

enas = ENAS(inputShape=torch.Size([4, 3, 32, 32]))
enas.construct()

sample = [1, 3, 1, 1, 1, 7, 1, 1, 1, 0, 4, 0, 4, 0, 0, 0, 0]
enas.sampleArchitecture(sample)

# Render the graph
renderGraph(enas.graphHandler, os.path.join(currentdir, 'Temp'))

train_images, train_labels, test_images, test_labels = load_cifar10(data_dir=os.path.join(currentdir, 'Datasets/CIFAR-10'), batch_size=256)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

enas.sample.to(device)

untrained_accuracy = test_network(enas.sample, test_images, test_labels, device)
print(f'Accuracy of the untrained network on the test images: {untrained_accuracy}%')

start_time = time.time()
train_network(enas.sample, train_images, train_labels, test_images, test_labels, device, num_epochs=100)
end_time = time.time()
elapsed_time = end_time - start_time

trained_accuracy = test_network(enas.sample, test_images, test_labels, device)
print(f'Accuracy of the trained network on the test images: {trained_accuracy}%')
print(f'Training time: {elapsed_time}')
    
if trained_accuracy > 90:
    print("\U0001F60E")
