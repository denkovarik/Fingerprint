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
        
        print("CIFAR-10 training and test sets are loaded and available in ", dataDir)
        
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
    
def trainNetwork(model: CustomCNN, trainloader: CIFAR10DataLoader, testloader: CIFAR10DataLoader, device: PythonObject, maxEpochs: Int) -> Float64:
    var nn = Python.import_module("torch.nn")
    var optim = Python.import_module("torch.optim")
    
    var criterion = nn.CrossEntropyLoss()    
    var optimizer = optim.Adam(model.registerParameters(), lr=0.001)
    var highValAcc: Float64 = 0
    var highValAccCnt: Int = 0
    var improving: Bool = True
    var epoch: Int = 0
    while improving:
        var running_loss: PythonObject = 0.0
        var cnt: Float64 = 0.0
        for i in range(trainloader.__len__()):
            var images = trainloader.batchImages[i].to(device)
            var labels = trainloader.batchLabels[i].to(device)
           
            optimizer.zero_grad()
            var outputs: PythonObject = model.forward(images)
            var loss: PythonObject = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss = running_loss + loss.item()
        
        #print("Epoch " + str(epoch + 1) + " Loss: " + str(running_loss / trainloader.__len__()))
        
        # Test
        var accuracy: Float64 = testNetwork(model=model, testloader=testloader, device=device)
        #print("Validation Accuracy: " + str(accuracy) + "%")
        
        if accuracy > highValAcc:
            highValAcc = accuracy
            highValAccCnt = 0
        else:
            highValAccCnt = highValAccCnt + 1
            
        epoch = epoch + 1
            
        if highValAccCnt > 2 or epoch >= maxEpochs:
            improving = False
        
    return highValAcc

    
def main():
    var nn = Python.import_module("torch.nn")
    var optim = Python.import_module("torch.optim")
    var torch = Python.import_module("torch")
    var torchvision = Python.import_module("torchvision")
    var transforms = Python.import_module("torchvision.transforms")   
    var time = Python.import_module("time")
    var random = Python.import_module("random")
    
    var dataDir: String = "Datasets/CIFAR-10"
    var num_epochs: Int = 15
    var batchSize: Int = 512
    var inputShape: PythonObject = torch.Size([batchSize, 3, 32, 32])
    
    # Construct the ENAS
    var enas = ENAS(inputShape)
    enas.construct()
    
    var sampleGraph = enas.graphHandler.getRandomSampleArchitecture()    
    
    # Init population
    var popSize: Int = 100
    var population: List[Graph] = List[Graph](sampleGraph)
        
    while(population.__len__() < popSize):
        sampleGraph = enas.graphHandler.getRandomSampleArchitecture()    
        population.append(sampleGraph)
    
    var cpu: PythonObject = torch.device("cpu")
    var device: PythonObject = torch.device("cpu")
    var cuda_available = torch.cuda.is_available()
    if cuda_available:
        device = torch.device("cuda") 
    print("Using device " + str(device))
    
      
    # Load cifar10 dataset
    var trainloader = CIFAR10DataLoader(dataDir=dataDir, batchSize=batchSize, device=device, train=True, shuffle=True)
    var testloader = CIFAR10DataLoader(dataDir=dataDir, batchSize=batchSize, device=device, train=False, shuffle=True)
           
    var startTime = time.perf_counter()
    var maxEpochs: Int = 1
    
    var accuracies: List[Float64] = List[Float64]()
    var bestInd: Int = 0
    var generation: Int = 0
    
    while True:
        var genbest: Float64 = 0.0
        var accuracies: List[Float64] = List[Float64]()
        
        for i in range(len(population)):
            var sampleNetwork = CustomCNN(population[i], inputShape)
            sampleNetwork.initSubweights()
            sampleNetwork.to(device=device) 
            var accuracy: Float64 = trainNetwork(model=sampleNetwork, trainloader=trainloader, testloader=testloader, device=device, maxEpochs=maxEpochs)
            if accuracy > genbest:
                genbest = accuracy
                bestInd = i
            accuracies.append(accuracy)
            sampleNetwork.to(device=cpu) 
            
        print("Generation " + str(generation) + " Best Validation Accuracy: " + str(genbest) + "%  => [", end="")
        for i in range(population[bestInd].sampleArch.__len__()):
            print(population[bestInd].sampleArch[i], end="")
            if i < population[bestInd].sampleArch.__len__() - 1:
                print(", ", end="")
        print("]")
        
        # Trim population
        var suvivors: List[Graph] = List[Graph]()
        var thres = popSize / 2
        var best: Float64 = 0.0
        var bestInd: Int = 0    
        
        while thres > 0.0:
            best = 0.0
            bestInd = 0    
            for i in range(len(accuracies)):
                if accuracies[i] > best:
                    best = accuracies[i]
                    bestInd = i
            
            survivor = enas.graphHandler.getSampleArchitecture(population[bestInd].sampleArch)    
            suvivors.append(survivor)
                 
            population.pop(bestInd)
            accuracies.pop(bestInd)
            thres = thres - 1.0
            
        population = List[Graph](suvivors[0])        
        
        # Reproduce
        var i: Int = 0
        var survivorsLen = len(suvivors)
        # Mutate
        while i < survivorsLen:
            var ind: Int = random.randint(0, suvivors[i].sampleArch.__len__() - 1)
            var dir: Int = random.randint(0, 1)
            var mutantArch: List[Int] = List[Int]()
            for j in range(suvivors[i].sampleArch.__len__()):
                if j == ind:
                    if dir == 0 and suvivors[i].sampleArch[j] > 0: 
                        var mInd: Int = suvivors[i].sampleArch[j] - 1
                        mutantArch.append(mInd)
                    else: 
                        var mInd: Int = suvivors[i].sampleArch[j] + 1
                        mutantArch.append(mInd)
                else:
                    mutantArch.append(suvivors[i].sampleArch[j])
            var mutant: Graph = enas.graphHandler.getSampleArchitecture(mutantArch)
            population.append(mutant)
            i = i + 1

        while(population.__len__() < popSize):
            sampleGraph = enas.graphHandler.getRandomSampleArchitecture()    
            population.append(sampleGraph)
                
        generation = generation + 1
        if generation % 10 == 0:
            maxEpochs = maxEpochs + 1
    
    var endTime = time.perf_counter()
    
    
    
    print("Training completed in " + str(endTime - startTime) + " seconds")
        
    print('Finished Training')