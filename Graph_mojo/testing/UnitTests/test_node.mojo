from testing import assert_equal, assert_not_equal, assert_true, assert_false
from python import Python, PythonObject
from structs.Nodes import NodeType, NormalizationType, PoolingType, ActivationType
from structs.Nodes import Node, InputNode, OutputNode, NormalizationNode, PoolingNode, ActivationNode, FlattenNode
from structs.Nodes import ConvolutionalNode, LinearNode, NodeTrait


def test_execution():
    # Just tests running a mojo test
    assert_equal(0, 0)

def test_nodeTypes():
    """
    Tests the ability to use NodeType.
    """	    
    # INPUT
    var nodeType = NodeType.INPUT
    assert_equal(nodeType.value, NodeType.INPUT.value)
        
    # OUTPUT
    nodeType = NodeType.OUTPUT
    assert_equal(nodeType.value, NodeType.OUTPUT.value)
        
    # CONVOLUTION
    nodeType = NodeType.CONVOLUTION
    assert_equal(nodeType.value, NodeType.CONVOLUTION.value)
        
    # NORMALIZATION
    nodeType = NodeType.NORMALIZATION
    assert_equal(nodeType.value, NodeType.NORMALIZATION.value)
        
    # POOLING
    nodeType = NodeType.POOLING
    assert_equal(nodeType.value, NodeType.POOLING.value)
        
    # FLATTEN
    nodeType = NodeType.FLATTEN
    assert_equal(nodeType.value, NodeType.FLATTEN.value)
        
    # LINEAR
    nodeType = NodeType.LINEAR
    assert_equal(nodeType.value, NodeType.LINEAR.value)
            
    # ACTIVATION
    nodeType = NodeType.ACTIVATION
    assert_equal(nodeType.value, NodeType.ACTIVATION.value)
    
def test_normalizationTypes():
    """
    Tests the ability to use NormalizationType.
    """	    
    # NO_NORM
    var normalizationType = NormalizationType.NO_NORM
    assert_equal(normalizationType.value, NormalizationType.NO_NORM.value)
        
    # BATCH_NORM
    normalizationType = NormalizationType.BATCH_NORM
    assert_equal(normalizationType.value, NormalizationType.BATCH_NORM.value)
    
def test_poolingTypes():
    """
    Tests the ability to use PoolingType.
    """	    
    # NO_POOLING
    var poolingType = PoolingType.NO_POOLING
    assert_equal(poolingType.value, PoolingType.NO_POOLING.value)
        
    # MAX_POOLING
    poolingType = PoolingType.MAX_POOLING
    assert_equal(poolingType.value, PoolingType.MAX_POOLING.value)
    
def test_activationTypes():
    """
    Tests the ability to use ActivationType.
    """	    
    # RELU
    var activationType = ActivationType.RELU
    assert_equal(activationType.value, ActivationType.RELU.value)
        
    # NONE
    activationType = ActivationType.LINEAR
    assert_equal(activationType.value, ActivationType.LINEAR.value)
    
def test_inputNode():
    """
    Tests the ability to construct and use the InputNode struct.
    """
    torch = Python.import_module("torch")

    var node = InputNode(inputShape=torch.Size([4, 3, 32, 32]))
    assert_equal(node.numChannels, 3)
    assert_equal(node.name, 'input')
    assert_equal(node.displayName, 'Input(numChannels=3)')
    
def test_inputNodeWrapper():
    """
    Tests the ability to construct and use the Node struct wrapper for InputNode struct.
    """
    torch = Python.import_module("torch")

    node = Node(theNode=InputNode(inputShape=torch.Size([4, 3, 32, 32])))
    assert_equal(node.node[InputNode].numChannels, 3)
    assert_equal(node.node[InputNode].name, 'input')
    assert_equal(node.node[InputNode].displayName, 'Input(numChannels=3)')
    
def test_OutputNode():
    """
    Tests the ability to construct and use the OutputNode stuct.
    """
    node = OutputNode()
    assert_equal(node.name, 'output')
    assert_equal(node.displayName, 'Output')
    
def test_OutputNodeWrapper():
    """
    Tests the ability to construct and use the Node struct wrapper for OutputNode struct.
    """
    node = Node(theNode=OutputNode())
    assert_equal(node.node[OutputNode].name, 'output')
    assert_equal(node.node[OutputNode].displayName, 'Output')
    
def test_NormalizationNodeWrapperConstruction():
    """
    Tests the ability to construct and use a node of the NormalizationNode wrapper (Node) struct.
    """
    uuid = Python.import_module("uuid")
    pytorchLayerId = uuid.uuid4()
    node = Node(theNode=NormalizationNode(name="name", 
                             normalizationType=NormalizationType.BATCH_NORM, 
                             numFeatures=12, pytorchLayerId=pytorchLayerId))
    assert_equal(node.node[NormalizationNode].name, 'name')
    assert_equal(node.node[NormalizationNode].displayName, 'Batch Normalization')
    assert_equal(node.node[NormalizationNode].normalizationType.value, NormalizationType.BATCH_NORM.value)
    assert_equal(pytorchLayerId, node.node[NormalizationNode].pytorchLayerId)
    
def test_forwardNoNormalization():
    """
    Test forward propigation for the NormalizationNode wrapper (Node) struct with no normalization set.
    """
    torch = Python.import_module("torch")
    nn = Python.import_module("torch.nn")
    uuid = Python.import_module("uuid")
    pytorchLayerId = uuid.uuid4()
    
    node = Node(theNode=NormalizationNode(name="name", 
                         normalizationType=NormalizationType.NO_NORM, 
                         numFeatures=3, pytorchLayerId=pytorchLayerId))
    
    var inputTensor: PythonObject = torch.randn(1, 3, 5, 5)
    
    var nodeNoNormTestOutput = node.forward(inputTensor)
    assert_true(torch.allclose(inputTensor, nodeNoNormTestOutput))

def test_forwardBatchNormalization():
    """
    Test forward propigation for the NormalizationNode wrapper (Node) struct with no normalization set.
    """
    torch = Python.import_module("torch")
    nn = Python.import_module("torch.nn")
    uuid = Python.import_module("uuid")
    pytorchLayerId = uuid.uuid4()

    node = Node(theNode=NormalizationNode(name="name", 
                             normalizationType=NormalizationType.BATCH_NORM, 
                             numFeatures=3, pytorchLayerId=pytorchLayerId))
    
    var inputTensor: PythonObject = torch.randn(1, 3, 5, 5)
    var batchNormModule: PythonObject = nn.BatchNorm2d(3) 
    
    var batchNormModuleOuptput = batchNormModule(inputTensor)
    var nodeBatchNormTestOutput = node.forward(inputTensor)
    assert_false(torch.allclose(inputTensor, nodeBatchNormTestOutput))
    assert_true(torch.allclose(batchNormModuleOuptput, nodeBatchNormTestOutput))
    
def test_toStringNoNorm():
    """
    Tests the NormalizationNode wrapper (Node) struct to string overloaded function for NO_NORM type.
    """
    uuid = Python.import_module("uuid")
    pytorchLayerId = uuid.uuid4()

    node = Node(theNode=NormalizationNode(name="name", 
                         normalizationType=NormalizationType.NO_NORM, 
                         numFeatures=3, pytorchLayerId=pytorchLayerId))
                  
    assert_equal(node.node[NormalizationNode].__str__(), 'NoNorm2d()')

def test_toStringBatchNorm():
    """
    Tests the NormalizationNode wrapper (Node) struct to string overloaded function for NO_NORM type.
    """
    uuid = Python.import_module("uuid")
    pytorchLayerId = uuid.uuid4()
    nn = Python.import_module("torch.nn")

    node = Node(theNode=NormalizationNode(name="name", 
                         normalizationType=NormalizationType.BATCH_NORM, 
                         numFeatures=3, pytorchLayerId=pytorchLayerId))
    var batchNormModule: PythonObject = nn.BatchNorm2d(3)
                  
    assert_equal(node.node[NormalizationNode].__str__(), str(batchNormModule))
    
def test_forwardNoPooling():
    """
    Test forward propigation for the PoolingNode struct with no pooling set.
    """
    torch = Python.import_module("torch")
    nn = Python.import_module("torch.nn")
    uuid = Python.import_module("uuid")
    pytorchLayerId = uuid.uuid4()
    
    node = Node(PoolingNode(name="name", 
                             poolingType=PoolingType.NO_POOLING))
    
    var inputTensor: PythonObject = torch.randn(1, 3, 5, 5)
    
    var nodeNoPoolingTestOutput =  node.forward(inputTensor)
    assert_true(torch.allclose(inputTensor, nodeNoPoolingTestOutput))

def test_forwardMaxPooling():
    """
    Test forward propigation for the PoolingNode struct with no pooling set.
    """
    torch = Python.import_module("torch")
    nn = Python.import_module("torch.nn")
    uuid = Python.import_module("uuid")
    pytorchLayerId = uuid.uuid4()

    node = Node(PoolingNode(name="name", 
                             poolingType=PoolingType.MAX_POOLING))
    
    var inputTensor: PythonObject = torch.randn(1, 3, 8, 8)
    var maxPoolingModule: PythonObject = nn.MaxPool2d(kernel_size=2, stride=2) 
    
    var maxPoolingModuleOuptput = maxPoolingModule(inputTensor)
    var nodeMaxPoolingTestOutput = node.forward(inputTensor)
    assert_false(inputTensor.shape == nodeMaxPoolingTestOutput.shape)
    assert_true(torch.allclose(maxPoolingModuleOuptput, nodeMaxPoolingTestOutput))
    
def test_toStringNoPooling():
    """
    Tests the to string overloaded function for NO_NORM type.
    """
    uuid = Python.import_module("uuid")
    pytorchLayerId = uuid.uuid4()

    node = Node(PoolingNode(name="name", 
                             poolingType=PoolingType.NO_POOLING))
                  
    assert_equal(node.node[PoolingNode].__str__(), 'NoPooling2d()')

def test_toStringMaxPooling():
    """
    Tests the to string overloaded function for MAX_POOLING type.
    """
    uuid = Python.import_module("uuid")
    pytorchLayerId = uuid.uuid4()
    nn = Python.import_module("torch.nn")

    node = Node(PoolingNode(name="name", 
                             poolingType=PoolingType.MAX_POOLING))
    
    var maxPoolingModule: PythonObject = nn.MaxPool2d(kernel_size=2, stride=2) 
                  
    assert_equal(node.node[PoolingNode].__str__(), str(maxPoolingModule))
    
def test_ActivationNode():
    """
    Tests the ability to construct and use a node of the ActivationNode struct.
    """
    node = Node(ActivationNode(name='name', activationType=ActivationType.RELU))
    assert_equal(node.node[ActivationNode].name, 'name')
    assert_equal(node.node[ActivationNode].displayName, 'Relu Activation')
    assert_equal(node.node[ActivationNode].activationType.value, ActivationType.RELU.value)
    
def test_forwardLinearActivation():
    """
    Test forward propigation for the ActivationNode struct with linear activation set.
    """
    torch = Python.import_module("torch")
    nn = Python.import_module("torch.nn")
    uuid = Python.import_module("uuid")
    pytorchLayerId = uuid.uuid4()
    
    node = Node(ActivationNode(name='name', activationType=ActivationType.LINEAR))
    
    var inputTensor: PythonObject = torch.randn(1, 3, 5, 5)
    
    var linearActivationTestOutput = node.forward(inputTensor)
    assert_true(torch.allclose(inputTensor, linearActivationTestOutput))

def test_forwardReluActivation():
    """
    Test forward propigation for the ActivationNode struct with relu activation set.
    """
    torch = Python.import_module("torch")
    nn = Python.import_module("torch.nn")
    uuid = Python.import_module("uuid")
    pytorchLayerId = uuid.uuid4()

    node = Node(ActivationNode(name='name', activationType=ActivationType.RELU))
    
    var inputTensor: PythonObject = torch.randn(1, 3, 8, 8)
    var reluModule: PythonObject = nn.ReLU()
    
    var reluModuleOuptput = reluModule(inputTensor)
    var nodeReluActivationTestOutput = node.forward(inputTensor)
    assert_false(torch.allclose(inputTensor, nodeReluActivationTestOutput))
    assert_true(torch.allclose(reluModuleOuptput, nodeReluActivationTestOutput))
    
def test_toStringLinearActivation():
    """
    Tests the to string overloaded function for LINEAR activation type.
    """
    uuid = Python.import_module("uuid")
    pytorchLayerId = uuid.uuid4()

    node = Node(ActivationNode(name='name', activationType=ActivationType.LINEAR))
                  
    assert_equal(node.node[ActivationNode].__str__(), 'LinearActivation()')

def test_toStringReluActivation():
    """
    Tests the to string overloaded function for RELU type.
    """
    uuid = Python.import_module("uuid")
    pytorchLayerId = uuid.uuid4()
    nn = Python.import_module("torch.nn")

    node = Node(ActivationNode(name='name', activationType=ActivationType.RELU))
    var reluModule: PythonObject = nn.ReLU()
                  
    assert_equal(node.node[ActivationNode].__str__(), str(reluModule))

def test_FlattenNode():
    """
    Tests the ability to construct and use a node of the FlattenNode struct.
    """
    node = FlattenNode(name='name')
    assert_equal(node.name, 'name')
    assert_equal(node.displayName, 'Flatten')

def test_CovolutionalNode():
    """
    Tests the ability to construct and use a node of the CovolutionalNode struct.
    """
    # Valid Construction with int for kernel size
    uuid = Python.import_module("uuid")
    var pytorchLayerId = uuid.uuid4()
    node = ConvolutionalNode(name='name', kernel_size=3, 
                             maxNumInputChannels=128, 
                             maxNumOutputChannels=128, 
                             numOutputChannels=32,
                             layer=0, pytorchLayerId=pytorchLayerId)
    assert_true(node.name == 'name')
    assert_true(node.layer == 0)
    assert_true(node.pytorchLayerId == pytorchLayerId)
    assert_true(node.displayName == '3x3 Conv(oc=32)')
    assert_true(node.kernel_size == 3)
    assert_true(node.maxNumInputChannels == 128)
    assert_true(node.numOutputChannels == 32)

def test_LinearNode():
    """
    Tests the ability to construct and use a node of the LinearNode struct.
    """
    uuid = Python.import_module("uuid")
    pytorchLayerId = uuid.uuid4()
    node = LinearNode(name='name', 
                      maxNumInFeatures=512, 
                      maxNumOutFeatures=512,
                      numOutFeatures=32, 
                      layer=1, pytorchLayerId=pytorchLayerId)
    assert_true(node.name == 'name')
    assert_true(node.layer == 1)
    assert_true(node.pytorchLayerId == pytorchLayerId)
    assert_true(node.maxNumInFeatures == 512)
    assert_true(node.numOutFeatures == 32)
    assert_true(node.displayName == 'Linear(of=32)')   
