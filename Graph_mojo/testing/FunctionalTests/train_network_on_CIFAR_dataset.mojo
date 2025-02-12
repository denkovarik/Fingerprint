from python import Python, PythonObject
from structs.ENAS import ENAS
from structs.Graph import GraphHandler, Graph


def main():
    # Load cifar10 dataset
    var nn = Python.import_module("torch.nn")
    var optim = Python.import_module("torch.optim")
    var torch = Python.import_module("torch")
    var torchvision = Python.import_module("torchvision")
    var transforms = Python.import_module("torchvision.transforms")
    var data_dir: String = "Datasets/CIFAR-10"
    
    var batchSize: Int = 512
    var inputShape: PythonObject = torch.Size([batchSize, 3, 32, 32])
    
    # Define transformations
    var transform = transforms.Compose([
        transforms.ToTensor(),
        #transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    # Load or download the trainset
    trainset = torchvision.datasets.CIFAR10(root=data_dir, train=True, download=True, transform=transform)
    
    # Load or download the testset
    testset = torchvision.datasets.CIFAR10(root=data_dir, train=False, download=True, transform=transform)
    
    print("CIFAR-10 training and test sets are loaded and available in ", data_dir)
    
    # Construct the ENAS
    var enas = ENAS(inputShape)
    enas.construct()
    
    var sample: List[Int] = List(1, 0, 1, 1, 0, 5, 1, 1, 1, 0, 3, 1, 1, 1, 0, 0, 0)
    enas.sampleArchitecture(sample)
    
    enas.sample.initSubweights()
    
    var device: PythonObject = torch.device("cpu")
    var cuda_available = torch.cuda.is_available()
    if cuda_available:
        device = torch.device("cuda") 
    print("Using device " + str(device))
    
    enas.sample.to(device=device)
    
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batchSize, shuffle=True)
    var criterion = nn.CrossEntropyLoss()    
    var optimizer = optim.Adam(enas.sample.registerParameters(), lr=0.001)
    
    var num_epochs: Int = 3
    
    for epoch in range(num_epochs):
        var running_loss: PythonObject = 0.0
        var tot: Float64 = trainloader.__len__()
        var cnt: Float64 = 0.0
        for batch in trainloader:
            var images: PythonObject = batch[0]
            var labels: PythonObject = batch[1]
            
            images = images.pin_memory()
            labelss = labels.pin_memory()

            images = images.to(device)
            labels = labels.to(device)
           
            optimizer.zero_grad()
            var outputs: PythonObject = enas.sample.forward(images)
            var loss: PythonObject = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss = running_loss + loss.item()
            
            cnt = cnt + 1
            
            if cnt % 20 == 0:
                var com: Float64 = 100.0 * cnt / tot
                print(com)
        
        #print("Epoch " + str(epoch + 1))
        print("Epoch " + str(epoch + 1) + " Loss: " + str(running_loss / len(trainloader)))
        
    print('Finished Training')