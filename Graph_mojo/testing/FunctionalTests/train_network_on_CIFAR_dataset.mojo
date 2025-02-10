from python import Python, PythonObject
from structs.ENAS import ENAS
from structs.Graph import GraphHandler, Graph


def main():
    # Load cifar10 dataset
    torch = Python.import_module("torch")
    torchvision = Python.import_module("torchvision")
    transforms = Python.import_module("torchvision.transforms")
    var data_dir: String = "Datasets/CIFAR-10"
    
    # Define transformations
    var transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    # Load or download the trainset
    trainset = torchvision.datasets.CIFAR10(root=data_dir, train=True, download=True, transform=transform)
    
    # Load or download the testset
    testset = torchvision.datasets.CIFAR10(root=data_dir, train=False, download=True, transform=transform)
    
    print("CIFAR-10 training and test sets are loaded and available in ", data_dir)
    
    # Construct the ENAS
    var enas = ENAS()
    enas.construct()
    
    var sample: List[Int] = List(1, 0, 1, 1, 0, 5, 1, 1, 1, 0, 3, 1, 1, 1, 0, 0, 0)
    enas.sampleArchitecture(sample)
    
    #var device: PythonObject = torch.device("cpu")
    #var cuda_available = torch.cuda.is_available()
    #if cuda_available:
    #    device = torch.device("cuda") 

    #enas.sample.to(device=device)
