import cupy
import numpy

def np_get(arr):  # Todo: to be deleted(after checking)
    if isinstance(arr, cupy.ndarray):
        return arr.get()
    elif isinstance(arr, numpy.ndarray):
        return arr
    else:
        raise TypeError(f"np_get: Unsupported array type {type(arr)}")