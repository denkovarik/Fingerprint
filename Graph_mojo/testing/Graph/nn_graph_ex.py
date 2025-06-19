from dataclasses import dataclass

import time
import numpy as np
from max.dtype import DType
from max.graph import Graph, TensorType, TensorValue, ops, DeviceRef
from max.driver import CPU, Accelerator, Tensor, accelerator_count
from max.engine import InferenceSession


@dataclass
class Linear:
    weight: np.ndarray
    bias: np.ndarray

    def __call__(self, x: TensorValue) -> TensorValue:
        weight_tensor = ops.constant(self.weight, dtype=DType.float32, device=DeviceRef.CPU())
        bias_tensor = ops.constant(self.bias, dtype=DType.float32, device=DeviceRef.CPU())
        return ops.matmul(x, weight_tensor) + bias_tensor


device = CPU() #if accelerator_count() == 0 else Accelerator()

linear_graph = Graph(
    "linear",
    Linear(np.ones((2, 2)), np.ones((2,))),
    input_types=[
        TensorType(
            DType.float32, (2,),
            device=DeviceRef.from_device(device)
        )
    ]
)

# Set up an inference session for running the graph.
session = InferenceSession(devices=[device])

# Compile the graph.
model = session.load(linear_graph)