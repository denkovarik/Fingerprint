from pathlib import Path
import numpy as np
import time
import torch
import torch.nn as nn

from max.driver import CPU, Accelerator, Tensor, accelerator_count
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import DeviceRef, Graph, TensorType, ops


num_itrs = 10000

# Create a 2x2 matrix
matrix = np.random.uniform(size=(64, 1, 28, 28)).astype(np.float32)
matrix2 = np.random.uniform(size=(784, 10)).astype(np.float32)

# Pytorch Test
print("Pytorch Speed Tests")
pytorch_device = torch.device("cpu")
#pytorch_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {pytorch_device}")

# Flatten Speed Test
tm1_pytorch = time.time()

t1 = torch.from_numpy(matrix).to(pytorch_device)
t2 = torch.from_numpy(matrix2).to(pytorch_device)

flatten = nn.Flatten()

for i in range(num_itrs):
    t1_flat = flatten(t1)
    
dur1_pytorch = time.time() - tm1_pytorch
print(f"{num_itrs} iterations for Pytorch Flatten method took {dur1_pytorch} seconds")

# Matrix Multiplication Speed Test
t1_flat = flatten(t1)

tm2_pytorch = time.time()

for i in range(num_itrs):
    result = t1_flat @ t2
    
dur2_pytorch = time.time() - tm2_pytorch
print(f"{num_itrs} iterations for Pytorch Matrix Multiplication method took {dur2_pytorch} seconds")

print(result.shape)




# Max Graph Test
print("\nMax Graph Speed Tests")
mojo_kernels = Path(__file__).parent / "operations"
dtype = DType.float32
device = CPU()
#device = CPU() if accelerator_count() == 0 else Accelerator()
print(f"Device: {device}")

# Max Graph Flatten Speed Test
tm1 = time.time()

tensor = Tensor.from_numpy(matrix).to(device)

with Graph("flatten_demo") as graph:
    tensor = Tensor.from_numpy(matrix).to(device)
    
    for i in range(num_itrs):
        flattened_tensor = tensor.view(DType.float32, shape=([64, 784]))

dur1 = time.time() - tm1
    
print(f"{num_itrs} iterations for Max Graph Flatten method took {dur1} seconds")

# Max Graph Matrix Multiplication Speed Test

# Configure our simple one-operation graph.
out_values = np.random.uniform(size=(64, 10)).astype(np.float32)
out_tensor = Tensor.from_numpy(out_values).to(device)

with Graph(
    "matrix_multiplication",
    input_types=[
        TensorType(
            dtype,
            shape=[64, 784],
            device=DeviceRef.from_device(device),
        ),
        TensorType(
            dtype,
            shape=[784, 10],
            device=DeviceRef.from_device(device),
        ),
    ],
    custom_extensions=[mojo_kernels],
) as graph:
    # Take in the two inputs to the graph.
    lhs, rhs = graph.inputs
    output = ops.custom(
        name="matrix_multiplication",
        values=[lhs, rhs],
        out_types=[
            TensorType(
                dtype=lhs.tensor.dtype,
                shape=out_tensor.shape,
                device=DeviceRef.from_device(device),
            )
        ],
    )[0].tensor
    graph.output(output)

# Set up an inference session for running the graph.
session = InferenceSession(devices=[device])

# Compile the graph.
model = session.load(graph)

lhs_tensor = tensor.view(DType.float32, shape=([64, 784])).to(device)
rhs_tensor = Tensor.from_numpy(matrix2).to(device)

tm2 = time.time()
for i in range(num_itrs):
    result = model.execute(lhs_tensor, rhs_tensor)[0]

dur2 = time.time()-tm2
print(f"{num_itrs} iterations for Max Graph Matrix Multiplication method took {dur2} seconds")

result = model.execute(lhs_tensor, rhs_tensor)[0]
pytorch_result = t1_flat @ t2

isclose = np.all(np.isclose(result.to_numpy(), pytorch_result.numpy()))
print("Are the results close to each other?: %s" % (isclose))

