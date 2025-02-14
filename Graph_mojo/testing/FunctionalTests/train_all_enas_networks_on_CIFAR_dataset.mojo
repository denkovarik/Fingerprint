from python import Python, PythonObject
from structs.ENAS import ENAS, CustomCNN
from structs.Graph import GraphHandler, Graph


@value
struct CIFAR10DataLoader:
    var dataDir: String
    var batchSize: Int
    var transform: PythonObject
    var dataset: PythonObject
    var dataloader: PythonObject
    var device: PythonObject
    var batchImages: List[PythonObject]
    var batchLabels: List[PythonObject]
    var train: Bool
    var shuffle: Bool
    var dataLoaderLength: Int
    
    fn __init__(inout self, dataDir: String, batchSize: Int, device: PythonObject, train: Bool, shuffle: Bool) raises:
        var nn = Python.import_module("torch.nn")
        var optim = Python.import_module("torch.optim")
        var torch = Python.import_module("torch")
        var torchvision = Python.import_module("torchvision")
        var transforms = Python.import_module("torchvision.transforms")
        
        self.dataDir = dataDir
        self.batchSize = batchSize
        self.device = device
        self.batchImages = List[PythonObject]()
        self.batchLabels = List[PythonObject]()
        self.train = train
        self.shuffle = shuffle
    
        # Define transformations
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            #transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        
        # Load or download the dataset
        self.dataset = torchvision.datasets.CIFAR10(root=dataDir, train=train, download=True, transform=self.transform)
        
        #print("CIFAR-10 training and test sets are loaded and available in ", dataDir)
        
        self.dataloader = torch.utils.data.DataLoader(self.dataset, batch_size=self.batchSize, shuffle=shuffle)
        self.dataLoaderLength = len(self.dataloader)
     
        # Precompile all of the batches to avoid doing this in training loop
        for batch in self.dataloader:
            var images: PythonObject = batch[0]
            var labels: PythonObject = batch[1]            
            images = images.pin_memory()
            labels = labels.pin_memory()
            self.batchImages.append(images)
            self.batchLabels.append(labels)
            
    fn __len__(inout self) -> Int:
        return self.dataLoaderLength
        
        
def testNetwork(model: CustomCNN, testloader: CIFAR10DataLoader, device: PythonObject) -> Float64:
    var torch = Python.import_module("torch")

    var correct: Int = 0
    var total: Int = 0     

    for i in range(testloader.__len__()):
        var images: PythonObject = testloader.batchImages[i].to(device)
        var labels: PythonObject = testloader.batchLabels[i].to(device)
        var outputs: PythonObject = model.forward(images)          
        var rslt: List[PythonObject] = torch.max(outputs.data, 1)
        total = total + labels.__len__()
        for j in range(rslt[0][1].__len__()):
            if rslt[0][1][j].item() == labels[j].item():
                correct = correct +  1
     
    var accuracy: Float64 = 100.0 * correct / total
    return accuracy


def main():
    var nn = Python.import_module("torch.nn")
    var optim = Python.import_module("torch.optim")
    var torch = Python.import_module("torch")
    var torchvision = Python.import_module("torchvision")
    var transforms = Python.import_module("torchvision.transforms")   
    var time = Python.import_module("time")
    
    var dataDir: String = "Datasets/CIFAR-10"
    var num_epochs: Int = 1
    var batchSize: Int = 512
    var inputShape: PythonObject = torch.Size([batchSize, 3, 32, 32])
    
    var cpu: PythonObject = torch.device("cpu")
    var device: PythonObject = torch.device("cpu")
    var cuda_available = torch.cuda.is_available()
    if cuda_available:
        device = torch.device("cuda:1") 
    #print("Using device " + str(device))
      
    # Load cifar10 dataset
    var trainloader = CIFAR10DataLoader(dataDir=dataDir, batchSize=batchSize, device=device, train=True, shuffle=True)
    var testloader = CIFAR10DataLoader(dataDir=dataDir, batchSize=batchSize, device=device, train=False, shuffle=True)
        
    # Construct the ENAS
    var enas = ENAS(inputShape)
    enas.construct()

    var cnt = 0
    var incSampleArchitecture: Bool = True
    
    print("Validation Accuracy, Model")    
    
    # Iterate to the start
    var startArch: List[Int] = List[Int](1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0)
    #var stopArch: List[Int] = List[Int](1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0)
    
    var stop: Bool = False
    while stop == False and enas.graphHandler.sampleArchitecturesEnd == False:
        if enas.graphHandler.dfsStack == startArch:
            stop = True
        else:
            incSampleArchitecture = enas.graphHandler.incSampleArchitecture()   
    
    while incSampleArchitecture:    
        #if enas.graphHandler.dfsStack == stopArch:
        #    break
            
        var sample: List[Int] = List[Int](enas.graphHandler.dfsStack)
        
        enas.sampleArchitecture(sample) 
        enas.sample.to(device=device)
                                      
        var criterion = nn.CrossEntropyLoss()    
        var optimizer = optim.Adam(enas.sample.registerParameters(), lr=0.001)
        
        var startTime = time.perf_counter()
        for epoch in range(num_epochs):
            var running_loss: PythonObject = 0.0
            for i in range(trainloader.__len__()):
                var images = trainloader.batchImages[i].to(device)
                var labels = trainloader.batchLabels[i].to(device)
               
                optimizer.zero_grad()
                var outputs: PythonObject = enas.sample.forward(images)
                var loss: PythonObject = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
        
                running_loss = running_loss + loss.item()
            
            #print("Epoch " + str(epoch + 1) + " Loss: " + str(running_loss / trainloader.__len__()))
            
        # Test
        var accuracy: Float64 = testNetwork(model=enas.sample, testloader=testloader, device=device)            
        
        #print(str(100.0 * cnt / enas.graphHandler.numGraphSubnetworks), end=",")
        print(str(accuracy), end=",")
        
        
        print('"[', end="")
        for i in range(sample.__len__()):
            print(str(sample[i]), end="")
            if i < sample.__len__() - 1:
                print(",", end="")
        print(']"')
            
        var endTime = time.perf_counter()
        cnt = cnt + 1
        
        #print("Training completed in " + str(endTime - startTime) + " seconds")
        
        
        enas.sample.to(device=cpu)
        incSampleArchitecture = enas.graphHandler.incSampleArchitecture()
        
    print('Finished Training')