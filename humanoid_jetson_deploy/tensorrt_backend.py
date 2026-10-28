"""Explicit TensorRT 10 policy inference; importing this module needs no CUDA.

Only float32, batch-one 49/46 -> 12 policies are accepted. No serial imports.
Engine building is an offline operation, never part of the control loop.
"""
from __future__ import annotations

import ctypes as ct
import ctypes.util
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import threading
import time

import numpy as np


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class CudaRuntime:
    """Small checked libcudart ABI, with pinned host memory and one stream."""

    def __init__(self):
        candidates = [ctypes.util.find_library("cudart"), "libcudart.so",
                      "/usr/local/cuda/lib64/libcudart.so"]
        for candidate in filter(None, candidates):
            try:
                self.lib = ct.CDLL(candidate)
                break
            except OSError:
                pass
        else:
            raise RuntimeError("TensorRT backend requires installed libcudart")
        signatures = {
            "cudaGetErrorString": ([ct.c_int], ct.c_char_p),
            "cudaGetDevice": ([ct.POINTER(ct.c_int)], ct.c_int),
            "cudaRuntimeGetVersion": ([ct.POINTER(ct.c_int)], ct.c_int),
            "cudaDeviceGetAttribute": ([ct.POINTER(ct.c_int), ct.c_int, ct.c_int], ct.c_int),
            "cudaStreamCreate": ([ct.POINTER(ct.c_void_p)], ct.c_int),
            "cudaStreamSynchronize": ([ct.c_void_p], ct.c_int),
            "cudaStreamDestroy": ([ct.c_void_p], ct.c_int),
            "cudaMalloc": ([ct.POINTER(ct.c_void_p), ct.c_size_t], ct.c_int),
            "cudaFree": ([ct.c_void_p], ct.c_int),
            "cudaHostAlloc": ([ct.POINTER(ct.c_void_p), ct.c_size_t, ct.c_uint], ct.c_int),
            "cudaFreeHost": ([ct.c_void_p], ct.c_int),
            "cudaMemcpyAsync": ([ct.c_void_p, ct.c_void_p, ct.c_size_t, ct.c_int, ct.c_void_p], ct.c_int),
            "cudaEventCreate": ([ct.POINTER(ct.c_void_p)], ct.c_int),
            "cudaEventRecord": ([ct.c_void_p, ct.c_void_p], ct.c_int),
            "cudaEventElapsedTime": ([ct.POINTER(ct.c_float), ct.c_void_p, ct.c_void_p], ct.c_int),
            "cudaEventDestroy": ([ct.c_void_p], ct.c_int),
        }
        for name, (args, result) in signatures.items():
            fn = getattr(self.lib, name)
            fn.argtypes, fn.restype = args, result

    def call(self, name, *args):
        code = getattr(self.lib, name)(*args)
        if code:
            message = self.lib.cudaGetErrorString(code).decode("utf-8", "replace")
            raise RuntimeError(f"{name}: CUDA {code}: {message}")

    def pointer(self, name, *args):
        value = ct.c_void_p()
        self.call(name, ct.byref(value), *args)
        return value

    def identity(self):
        device, version, major, minor = (ct.c_int() for _ in range(4))
        self.call("cudaGetDevice", ct.byref(device))
        self.call("cudaRuntimeGetVersion", ct.byref(version))
        self.call("cudaDeviceGetAttribute", ct.byref(major), 75, device)
        self.call("cudaDeviceGetAttribute", ct.byref(minor), 76, device)
        # UUID is exposed by the driver API, not a libcudart symbol. Avoid
        # cudaDeviceProp's version-dependent struct layout when binding ctypes.
        driver = ct.CDLL(ctypes.util.find_library("cuda") or "libcuda.so.1")
        driver.cuInit.argtypes, driver.cuInit.restype = [ct.c_uint], ct.c_int
        driver.cuDeviceGet.argtypes, driver.cuDeviceGet.restype = [ct.POINTER(ct.c_int), ct.c_int], ct.c_int
        get_uuid = getattr(driver, "cuDeviceGetUuid_v2", None) or driver.cuDeviceGetUuid
        get_uuid.argtypes, get_uuid.restype = [ct.c_void_p, ct.c_int], ct.c_int
        driver_device = ct.c_int()
        uuid = (ct.c_ubyte * 16)()
        for name, fn, arguments in (("cuInit", driver.cuInit, (0,)),
                                    ("cuDeviceGet", driver.cuDeviceGet, (ct.byref(driver_device), device.value)),
                                    ("cuDeviceGetUuid", get_uuid, (uuid, driver_device.value))):
            code = fn(*arguments)
            if code:
                raise RuntimeError(f"{name}: CUDA driver error {code}")
        return {"uuid": bytes(uuid).hex(), "compute_capability": [major.value, minor.value],
                "cuda_runtime": version.value}


def build_engine(model_path, engine_path, obs_dim=49, workspace_mib=256):
    import tensorrt as trt
    cuda = CudaRuntime()
    identity = cuda.identity()
    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    # Strong typing preserves the ONNX float32 graph; no FP16/INT8 conversion.
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))
    parser = trt.OnnxParser(network, logger)
    if not parser.parse_from_file(str(Path(model_path).resolve())):
        raise RuntimeError("ONNX parse failed: " + "\n".join(
            str(parser.get_error(i)) for i in range(parser.num_errors)))
    if network.num_inputs != 1 or network.num_outputs != 1:
        raise ValueError("Policy must have one input and one output")
    inp, out = network.get_input(0), network.get_output(0)
    shape = tuple(inp.shape)
    if (obs_dim not in (46, 49) or inp.dtype != trt.float32 or out.dtype != trt.float32
            or len(shape) != 2 or shape[0] not in (-1, 1) or shape[1] != obs_dim
            or tuple(out.shape) not in ((1, 12), (-1, 12))):
        raise ValueError(f"Expected float32 [1,{obs_dim}] -> [1,12], got {shape} -> {tuple(out.shape)}")
    config = builder.create_builder_config()
    config.clear_flag(trt.BuilderFlag.TF32)
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, int(workspace_mib) * 1024**2)
    if shape[0] == -1:
        profile = builder.create_optimization_profile()
        profile.set_shape(inp.name, (1, obs_dim), (1, obs_dim), (1, obs_dim))
        config.add_optimization_profile(profile)
    started = time.perf_counter()
    plan = builder.build_serialized_network(network, config)
    if plan is None:
        raise RuntimeError("TensorRT engine build failed")
    path = Path(engine_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = bytes(plan)
    metadata = {"schema": 1, "model_sha256": sha256(model_path),
                "engine_sha256": hashlib.sha256(data).hexdigest(), "tensorrt": trt.__version__,
                "device": identity, "obs_dim": obs_dim, "precision": "float32_no_tf32",
                "build_seconds": time.perf_counter() - started}
    path.write_bytes(data)
    Path(str(path) + ".json").write_text(json.dumps(metadata, indent=2) + "\n")
    return metadata


class TensorRTSession:
    """ORT-compatible run boundary; returned actions are synchronized host copies.

    A single context and buffers are reused under a lock. profile=True adds CUDA
    events only for offline benchmarking; normal policy ticks have no events.
    """

    def __init__(self, model_path, engine_path, obs_dim=49, *, profile=False):
        if not engine_path:
            raise ValueError("TensorRT requires --trt-engine built by build_policy_engine.py")
        import tensorrt as trt
        self.cuda = CudaRuntime()
        self.stream = None
        self.device_buffers, self.host_buffers, self.events = [], [], []
        self.lock = threading.Lock()
        self.closed = False
        self.last_timing = {}
        self.profile = profile
        path = Path(engine_path)
        meta = json.loads(Path(str(path) + ".json").read_text())
        expected = {"schema": 1, "model_sha256": sha256(model_path),
                    "engine_sha256": sha256(path), "tensorrt": trt.__version__,
                    "device": self.cuda.identity(), "obs_dim": obs_dim,
                    "precision": "float32_no_tf32"}
        for key, value in expected.items():
            if meta.get(key) != value:
                raise ValueError(f"TensorRT engine metadata mismatch for {key}; rebuild on this device")
        self.metadata = meta
        self.logger = trt.Logger(trt.Logger.WARNING)
        self.runtime = trt.Runtime(self.logger)
        self.engine = self.runtime.deserialize_cuda_engine(path.read_bytes())
        if self.engine is None:
            raise RuntimeError("TensorRT engine deserialization failed")
        if self.engine.num_io_tensors != 2:
            raise ValueError("TensorRT policy must have exactly two IO tensors")
        inputs, outputs = [], []
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            if (self.engine.get_tensor_dtype(name) != trt.float32
                    or self.engine.get_tensor_location(name) != trt.TensorLocation.DEVICE
                    or self.engine.get_tensor_format(name) != trt.TensorFormat.LINEAR):
                raise ValueError("TensorRT policy requires linear float32 device IO")
            (inputs if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT else outputs).append(name)
        if len(inputs) != 1 or len(outputs) != 1:
            raise ValueError("TensorRT policy needs one input and one output")
        self.context = self.engine.create_execution_context()
        if self.context is None:
            raise RuntimeError("TensorRT context creation failed")
        self.input_name, self.output_name = inputs[0], outputs[0]
        if not self.context.set_input_shape(self.input_name, (1, obs_dim)):
            raise ValueError("TensorRT input does not accept batch-one observation")
        if (tuple(self.context.get_tensor_shape(self.input_name)) != (1, obs_dim)
                or tuple(self.context.get_tensor_shape(self.output_name)) != (1, 12)):
            raise ValueError("TensorRT policy IO shape mismatch")
        self.inputs = [SimpleNamespace(name=self.input_name, type="tensor(float)", shape=[1, obs_dim])]
        self.outputs = [SimpleNamespace(name=self.output_name, type="tensor(float)", shape=[1, 12])]
        try:
            self.stream = self.cuda.pointer("cudaStreamCreate")
            arrays = []
            for name, width in ((self.input_name, obs_dim), (self.output_name, 12)):
                size = width * 4
                device = self.cuda.pointer("cudaMalloc", size)
                self.device_buffers.append(device)
                host = self.cuda.pointer("cudaHostAlloc", size, 0)
                self.host_buffers.append(host)
                arrays.append(np.ctypeslib.as_array((ct.c_float * width).from_address(host.value)).reshape(1, width))
                if not self.context.set_tensor_address(name, device.value):
                    raise RuntimeError(f"Cannot bind TensorRT tensor {name}")
            self.host_input, self.host_output = arrays
            if profile:
                self.events = [self.cuda.pointer("cudaEventCreate") for _ in range(4)]
        except Exception:
            self.close()
            raise

    def get_inputs(self):
        return self.inputs

    def get_outputs(self):
        return self.outputs

    def run(self, output_names, feed):
        if output_names != [self.output_name] or set(feed) != {self.input_name}:
            raise ValueError("TensorRT policy requires its single named input/output")
        obs = np.asarray(feed[self.input_name])
        if obs.dtype != np.float32 or obs.shape != self.host_input.shape or not np.isfinite(obs).all():
            raise ValueError("TensorRT observation must be finite float32 with the exact policy shape")
        with self.lock:
            if self.closed:
                raise RuntimeError("TensorRT session is closed")
            start = time.perf_counter_ns()
            np.copyto(self.host_input, obs)
            def event(index):
                if self.profile:
                    self.cuda.call("cudaEventRecord", self.events[index], self.stream)
            event(0)
            self.cuda.call("cudaMemcpyAsync", self.device_buffers[0], self.host_buffers[0], obs.nbytes, 1, self.stream)
            event(1)
            if not self.context.execute_async_v3(stream_handle=self.stream.value):
                raise RuntimeError("TensorRT inference failed")
            event(2)
            self.cuda.call("cudaMemcpyAsync", self.host_buffers[1], self.device_buffers[1], self.host_output.nbytes, 2, self.stream)
            event(3)
            sync_start = time.perf_counter_ns()
            self.cuda.call("cudaStreamSynchronize", self.stream)
            sync_ms = (time.perf_counter_ns() - sync_start) / 1e6
            result = self.host_output.copy()
            if not np.isfinite(result).all():
                raise RuntimeError("TensorRT produced non-finite actions")
            timing = {"total_ms": (time.perf_counter_ns() - start) / 1e6, "host_sync_ms": sync_ms}
            if self.profile:
                for i, name in enumerate(("h2d_ms", "compute_ms", "d2h_ms")):
                    elapsed = ct.c_float()
                    self.cuda.call("cudaEventElapsedTime", ct.byref(elapsed), self.events[i], self.events[i + 1])
                    timing[name] = elapsed.value
            self.last_timing = timing
            return [result]

    def close(self):
        with self.lock:
            if self.closed:
                return
            self.closed = True
            if self.stream:
                self.cuda.call("cudaStreamSynchronize", self.stream)
            # Release context before its bound storage.
            self.context = None
            for event in self.events:
                self.cuda.call("cudaEventDestroy", event)
            for buffer in self.device_buffers:
                self.cuda.call("cudaFree", buffer)
            for buffer in self.host_buffers:
                self.cuda.call("cudaFreeHost", buffer)
            if self.stream:
                self.cuda.call("cudaStreamDestroy", self.stream)
            self.events, self.device_buffers, self.host_buffers = [], [], []
            self.stream = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass
