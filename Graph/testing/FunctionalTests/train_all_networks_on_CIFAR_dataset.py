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
from multiprocessing import Pool



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
        
        if self.high_val_acc_cnt > 3 or self.epoch >= self.max_epochs:
            self.improving = False
        
        return self.improving
    
    
def train_on_sample(sample, train_images, train_labels, test_images, test_labels, device):
    # Create a new model instance for each sample
    model = enas.sample
    model = train_network(model, train_images, train_labels, test_images, test_labels, device)
    return model
    
    
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
            
        validation_accuracy = test_network(enas.sample, test_images, test_labels, device)
        print(f'Epoch {epoch + 1}: Loss: {running_loss / num_batches:.4f}   Validation Accuracy: {validation_accuracy:.2f}%')

        epoch += 1
        
    print('Finished Training')
    return model
 

enas = ENAS(inputShape=torch.Size([4, 3, 32, 32]))
enas.construct()

train_images, train_labels, test_images, test_labels = load_cifar10(data_dir=os.path.join(currentdir, 'Datasets/CIFAR-10'), batch_size=16384)

cpu = torch.device("cpu")
device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

devices = []
for i in range(torch.cuda.device_count()):
    devices.append(f'cuda:{i}')

if len(devices) == 0:
    devices.append(f'cpu')

print(f"Using device: {devices}")

incSampleArchitecture = True

start_time = time.time()
while incSampleArchitecture == True:     
    for i in range(len(devices)):
        sample = enas.graphHandler.dfsStack
        enas.sampleArchitecture(sample) 
        device = torch.device(devices[i])
        enas.sample.to(device)
        print(sample)
        train_network(enas.sample, train_images, train_labels, test_images, test_labels, device, max_epochs=1000)
        enas.sample.to(device=cpu)
        incSampleArchitecture = enas.graphHandler.incSampleArchitecture()

end_time = time.time()
elapsed_time = end_time - start_time
