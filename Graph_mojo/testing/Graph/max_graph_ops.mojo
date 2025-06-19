#from max.driver import cpu, Accelerator, Tensor, accelerator_count
#from max.dtype import DType
#from max.engine import InferenceSession
from max.graph import Graph, TensorType, ops
#import time
from python import Python, PythonObject
from max import driver
from max.driver import Tensor

def main():
    var np = Python.import_module("numpy")
    var time = Python.import_module("time")
        
    print("Hello")
    
    var np_arr: PythonObject = np.zeros(Python.tuple(64, 1, 28, 28)).astype(np.float32)
    
    var cpu_tensor = driver.Tensor(shape=[2, 3], dtype=DType.float32)