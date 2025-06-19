from math import ceildiv
from algorithm import parallelize, vectorize
import compiler
from gpu import block_dim, block_idx, thread_idx
from gpu.host import DeviceContext
from runtime.asyncrt import DeviceContextPtr
from tensor import Tensor, TensorShape, InputTensor, ManagedTensorSlice, OutputTensor, foreach
from utils.index import Index, IndexList
from sys.info import simdwidthof, simdbitwidth

alias type = DType.float32
alias simd_width: Int = simdwidthof[type]()
alias simdwidth = simdwidthof[type]()



fn matrix_multiplication_cpu(
    out: ManagedTensorSlice[mut=True],
    lhs: ManagedTensorSlice[type = out.type, rank = out.rank],
    rhs: ManagedTensorSlice[type = out.type, rank = out.rank],
    ctx: DeviceContextPtr,
):    
    for i in range(out.shape()[0]): 
        for j in range(out.shape()[1]):
            out[i,j] = 0.0 
            
    @parameter
    fn calc_row(i: Int):
        for j in range(lhs.shape()[1]):
            for k in range(out.shape()[1]):
                out[i,k] += lhs[i,j] * rhs[j,k]

    parallelize[calc_row](out.shape()[0])
    

fn matrix_multiplication_gpu(
    out: ManagedTensorSlice[mut=True],
    lhs: ManagedTensorSlice[type = out.type, rank = out.rank],
    rhs: ManagedTensorSlice[type = out.type, rank = out.rank],
    ctx: DeviceContextPtr,
) raises:
    # Note: The following has not been tuned for any GPU hardware, and is an
    # instructional example for how a simple GPU function can be constructed
    # and dispatched.
    alias BLOCK_SIZE = 16
    var gpu_ctx = ctx.get_device_context()
    var vector_length = out.dim_size(0)

    # The function that will be launched and distributed across GPU threads.
    @parameter
    fn vector_addition_gpu_kernel(length: Int):
        var tid = block_dim.x * block_idx.x + thread_idx.x
        if tid < length:
            out[tid] = lhs[tid] + rhs[tid]

    # The vector is divided up into blocks, making sure there's an extra
    # full block for any remainder.
    var num_blocks = ceildiv(vector_length, BLOCK_SIZE)

    # The GPU function is compiled and enqueued to run on the GPU across the
    # 1-D vector, split into blocks of `BLOCK_SIZE` width.
    gpu_ctx.enqueue_function[vector_addition_gpu_kernel](
        vector_length, grid_dim=num_blocks, block_dim=BLOCK_SIZE
    )


@compiler.register("matrix_multiplication")
struct MatrixMultiplication:
    @staticmethod
    fn execute[
        # The kind of device this will be run on: "cpu" or "gpu"
        target: StaticString,
    ](
        # as num_dps_outputs=1, the first argument is the "output"
        out: OutputTensor,
        # starting here are the list of inputs
        lhs: InputTensor[type = out.type, rank = out.rank],
        rhs: InputTensor[type = out.type, rank = out.rank],
        # the context is needed for some GPU calls
        ctx: DeviceContextPtr,
    ) raises:
        # For a simple elementwise operation like this, the `foreach` function
        # does much more rigorous hardware-specific tuning. We recommend using
        # that abstraction, with this example serving purely as an illustration
        # of how lower-level functions can be used to program GPUs via Mojo.

        # At graph compilation time, we will know what device we are compiling
        # this operation for, so we can specialize it for the target hardware.
        @parameter
        if target == "cpu":
            matrix_multiplication_cpu(out, lhs, rhs, ctx)
        elif target == "gpu":
            pass
            #matrix_multiplication_gpu(out, lhs, rhs, ctx)
        else:
            raise Error("No known target:", target)
