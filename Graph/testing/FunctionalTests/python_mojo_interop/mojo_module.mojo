from python import PythonObject
from python.bindings import PythonModuleBuilder
import math
from os import abort

@export
fn PyInit_mojo_module() -> PythonObject:
    try:
        var m = PythonModuleBuilder("mojo_module")
        m.def_function[factorial]("factorial", docstring="Compute n!")
        return m.finalize()
    except e:
        return abort[PythonObject](String("error creating Python Mojo module:", e))

fn factorial(py_obj: PythonObject) raises -> PythonObject:
    # Raises an exception if `py_obj` is not convertible to a Mojo `Int`.
    var n = Int(py_obj)

    return math.factorial(n)
