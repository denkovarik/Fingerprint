import numpy as np

from max.dtype import DType
from max.graph import DeviceRef, Graph, TensorType, ops, TensorValue
from max.nn import Linear
from max.driver import CPU, Accelerator, Tensor, accelerator_count


device = CPU() #if accelerator_count() == 0 else Accelerator()

matrix = np.random.uniform(size=(64, 784)).astype(np.float32)
tensor = Tensor.from_numpy(matrix).to(device)
lhs_tensor = tensor.view(DType.float32, shape=([64, 784])).to(device)

in_dim = 784  # Input dimension from lhs_tensor
out_dim = 10  # Desired output dimension
has_bias = True
dtype = DType.float32

with Graph("linear_demo") as graph:
    tensor_val = ops.constant(matrix, dtype=DType.float32, device=DeviceRef.CPU())
    
    linear_layer = Linear(
        in_dim=in_dim,
        out_dim=out_dim,
        dtype=dtype,
        device=DeviceRef.CPU(),
        has_bias=has_bias,
        name="linear_layer"
    )

    output = linear_layer(tensor_val)

    print(output)
    
    
    