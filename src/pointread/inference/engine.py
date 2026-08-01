import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit  # noqa: F401  initializes CUDA context on import

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)


class Engine:
    """Thin wrapper around a serialized TensorRT engine.

    Handles buffer allocation and a single synchronous inference call.
    Knows nothing about hands or any specific model.
    """

    def __init__(self, path, input_shape, max_out=None):
        with open(path, "rb") as f, trt.Runtime(TRT_LOGGER) as rt:
            self.engine = rt.deserialize_cuda_engine(f.read())
        self.ctx = self.engine.create_execution_context()
        self.stream = cuda.Stream()
        self.in_name = None
        self.outputs = []
        for i in range(self.engine.num_io_tensors):
            n = self.engine.get_tensor_name(i)
            if self.engine.get_tensor_mode(n) == trt.TensorIOMode.INPUT:
                self.in_name = n
        self.ctx.set_input_shape(self.in_name, input_shape)

        self.buffers = {}
        for i in range(self.engine.num_io_tensors):
            n = self.engine.get_tensor_name(i)
            dt = trt.nptype(self.engine.get_tensor_dtype(n))
            shape = tuple(self.ctx.get_tensor_shape(n))
            shape = tuple(max_out if d < 0 else d for d in shape)
            host = cuda.pagelocked_empty(int(np.prod(shape)), dt)
            dev = cuda.mem_alloc(host.nbytes)
            self.ctx.set_tensor_address(n, int(dev))
            self.buffers[n] = {"host": host, "dev": dev, "shape": shape}
            if self.engine.get_tensor_mode(n) == trt.TensorIOMode.OUTPUT:
                self.outputs.append(n)

    def infer(self, x):
        b = self.buffers[self.in_name]
        np.copyto(b["host"], x.ravel())
        cuda.memcpy_htod_async(b["dev"], b["host"], self.stream)
        self.ctx.execute_async_v3(self.stream.handle)
        for n in self.outputs:
            cuda.memcpy_dtoh_async(self.buffers[n]["host"],
                                   self.buffers[n]["dev"], self.stream)
        self.stream.synchronize()
        out = {}
        for n in self.outputs:
            rt_shape = tuple(self.ctx.get_tensor_shape(n))
            flat = self.buffers[n]["host"]
            out[n] = flat[:int(np.prod(rt_shape))].reshape(rt_shape)
        return out
