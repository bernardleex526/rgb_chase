"""Inference runtimes: ONNX Runtime (PC / fallback) and TensorRT engines (Jetson)."""
from typing import Dict, List

import numpy as np


class OrtRunner:
    def __init__(self, path: str, providers=None):
        import onnxruntime as ort

        avail = ort.get_available_providers()
        prefs = providers or ["TensorrtExecutionProvider", "CUDAExecutionProvider", "CPUExecutionProvider"]
        self.session = ort.InferenceSession(path, providers=[p for p in prefs if p in avail])
        inp = self.session.get_inputs()[0]
        self.input_name = inp.name
        self.input_shape = inp.shape  # may contain symbolic dims

    def __call__(self, x: np.ndarray) -> List[np.ndarray]:
        return self.session.run(None, {self.input_name: x})


class TrtRunner:
    """TensorRT >= 10 engine with static input shape (build with trtexec, see README)."""

    def __init__(self, path: str):
        import tensorrt as trt
        import pycuda.autoinit  # noqa: F401  (creates the CUDA context)
        import pycuda.driver as cuda

        self._cuda = cuda
        logger = trt.Logger(trt.Logger.WARNING)
        with open(path, "rb") as f:
            self.engine = trt.Runtime(logger).deserialize_cuda_engine(f.read())
        self.context = self.engine.create_execution_context()
        self.stream = cuda.Stream()
        self.inputs: Dict[str, tuple] = {}
        self.outputs: Dict[str, tuple] = {}
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            shape = tuple(self.engine.get_tensor_shape(name))
            dtype = trt.nptype(self.engine.get_tensor_dtype(name))
            host = cuda.pagelocked_empty(int(np.prod(shape)), dtype)
            dev = cuda.mem_alloc(host.nbytes)
            self.context.set_tensor_address(name, int(dev))
            is_input = self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT
            (self.inputs if is_input else self.outputs)[name] = (host, dev, shape)
        self.input_name = next(iter(self.inputs))
        self.input_shape = self.inputs[self.input_name][2]

    def __call__(self, x: np.ndarray) -> List[np.ndarray]:
        host, dev, shape = self.inputs[self.input_name]
        np.copyto(host, x.ravel())
        self._cuda.memcpy_htod_async(dev, host, self.stream)
        self.context.execute_async_v3(self.stream.handle)
        outs = []
        for host_o, dev_o, shape_o in self.outputs.values():
            self._cuda.memcpy_dtoh_async(host_o, dev_o, self.stream)
            outs.append((host_o, shape_o))
        self.stream.synchronize()
        return [h.reshape(s).copy() for h, s in outs]


def make_runner(path: str):
    if path.endswith((".engine", ".plan", ".trt")):
        return TrtRunner(path)
    return OrtRunner(path)
