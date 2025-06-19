from random import seed

from max.engine import InferenceSession
from max.graph import Graph, TensorType
from max.tensor import Tensor, TensorShape


def main():
    graph = Graph(TensorType(DType.float32, "m", 2))
    # create a constant tensor value to later create a graph constant symbol
    var constant_value = Tensor[DType.float32](TensorShape(2, 2), 42.0)
    print("constant value:", constant_value)
    # create a constant symbol
    var constant_symbol = graph.constant(constant_value)
    # create a matmul node
    mm = graph[0] @ constant_symbol
    graph.output(mm)
    # verify
    graph.verify()

    # create session, load and compile the graph
    session = InferenceSession()
    model = session.load(graph)

    # generate random input
    seed(42)
    #var input0 = Tensor[DType.float32].rand((2, 2))
    #print("random 2x2 input0:", input0)
    #var ret = model.execute("input0", input0^)
    #print("matmul 2x2 result:", ret.get[DType.float32]("output0"))

    # with 3 x 2 matrix input
    #input0 = Tensor[DType.float32].randn((3, 2))
    #print("random 3x2 input0:", input0)
    #ret = model.execute("input0", input0^)
    #print("matmul 3x2 result:", ret.get[DType.float32]("output0"))