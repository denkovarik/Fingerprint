# ===----------------------------------------------------------------------=== #
# Copyright (c) 2025, Modular Inc. All rights reserved.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions:
# https://llvm.org/LICENSE.txt
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ===----------------------------------------------------------------------=== #

from pathlib import Path

import numpy as np
from max.driver import CPU, Accelerator, Tensor, accelerator_count
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import DeviceRef, Graph, TensorType, ops
import time
import torch

if __name__ == "__main__":
    mojo_kernels = Path(__file__).parent / "operations"

    vector_width = 100000000
    dtype = DType.float32

    # Place the graph on a GPU, if available. Fall back to CPU if not.
    device = CPU() if accelerator_count() == 0 else Accelerator()

    # Configure our simple one-operation graph.
    with Graph(
        "vector_addition",
        input_types=[
            TensorType(
                dtype,
                shape=[vector_width],
                device=DeviceRef.from_device(device),
            ),
            TensorType(
                dtype,
                shape=[vector_width],
                device=DeviceRef.from_device(device),
            ),
        ],
        custom_extensions=[mojo_kernels],
    ) as graph:
        # Take in the two inputs to the graph.
        lhs, rhs = graph.inputs
        output = ops.custom(
            name="vector_addition",
            values=[lhs, rhs],
            out_types=[
                TensorType(
                    dtype=lhs.tensor.dtype,
                    shape=lhs.tensor.shape,
                    device=DeviceRef.from_device(device),
                )
            ],
        )[0].tensor
        graph.output(output)

    # Set up an inference session for running the graph.
    session = InferenceSession(devices=[device])

    # Compile the graph.
    model = session.load(graph)

    # Fill input matrices with random values.
    lhs_values = np.random.uniform(size=(vector_width)).astype(np.float32)
    rhs_values = np.random.uniform(size=(vector_width)).astype(np.float32)
    
    # Convert to PyTorch tensors
    pytorch_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    tensor1 = torch.from_numpy(lhs_values)
    tensor2 = torch.from_numpy(rhs_values)
    tensor1 = tensor1.to(pytorch_device)
    tensor2 = tensor2.to(pytorch_device)

    # Create driver tensors from this, and move them to the accelerator.
    lhs_tensor = Tensor.from_numpy(lhs_values).to(device)
    rhs_tensor = Tensor.from_numpy(rhs_values).to(device)

    tm1 = time.time()

    # Perform the calculation on the target device.
    result = model.execute(lhs_tensor, rhs_tensor)[0]
    
    dur1 = time.time()-tm1
    
    tm2 = time.time()
    expected = lhs_values + rhs_values
    dur2 = time.time()-tm2
    
    tm3 = time.time()
    pytorch_result = torch.add(tensor1, tensor2)
    dur3 = time.time()-tm3
    
    # Copy values back to the CPU to be read.
    assert isinstance(result, Tensor)
    result = result.to(CPU())

    print("Left-hand-side values:")
    print(lhs_values)
    print()

    print("Right-hand-side values:")
    print(rhs_values)
    print()

    print("Graph result:")
    print(result.to_numpy())
    print()

    print("Expected result:")
    print(expected)

    isclose = np.all(np.isclose(result.to_numpy(), lhs_values + rhs_values))
    print("Are the results close to each other?: %s" % (isclose))
    
    print("\nVector Addition with Graph took ",dur1," seconds")
    print("\nVector Addition with Numpy took ",dur2," seconds")
    print("\nVector Addition with Pytorch took ",dur3," seconds")
