"""Inference boundary tests with a fake device; never import or open serial."""
import csv
import ctypes as ct
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from benchmark_policy_backends import load_observations, synthetic_smoke_observations
from policy_runner import HumanoidPolicy, observation_columns
from one_foot_policy import OneFootPolicy
import tensorrt_backend as backend


class FakeCuda:
    def __init__(self):
        self.memory = []
        self.pending = []
        self.calls = []

    def identity(self):
        return {"uuid": "test-device"}

    def pointer(self, name, *args):
        value = ct.create_string_buffer(args[0] if name in ("cudaMalloc", "cudaHostAlloc") else 8)
        self.memory.append(value)
        return ct.c_void_p(ct.addressof(value))

    def call(self, name, *args):
        self.calls.append(name)
        if name == "cudaMemcpyAsync":
            dst, src, size, kind, _ = args
            if kind == 1:
                ct.memmove(dst, src, size)
            else:
                self.pending.append((dst, src, size))
        elif name == "cudaStreamSynchronize":
            for dst, src, size in self.pending:
                ct.memmove(dst, src, size)
            self.pending.clear()


class FakeContext:
    def __init__(self, width=49):
        self.width = width
        self.addresses = {}
        self.executions = 0

    def set_input_shape(self, name, shape):
        return tuple(shape) == (1, self.width)

    def get_tensor_shape(self, name):
        return (1, self.width if name == "obs" else 12)

    def set_tensor_address(self, name, address):
        self.addresses[name] = address
        return True

    def execute_async_v3(self, stream_handle):
        self.executions += 1
        src = np.ctypeslib.as_array((ct.c_float * self.width).from_address(self.addresses["obs"]))
        dst = np.ctypeslib.as_array((ct.c_float * 12).from_address(self.addresses["actions"]))
        dst[:] = src[:12] * 2
        return True


class BackendTests(unittest.TestCase):
    def session(self, directory):
        model, engine = Path(directory)/"test.onnx", Path(directory)/"test.engine"
        model.write_bytes(b"model")
        engine.write_bytes(b"engine")
        meta = {"schema": 1, "model_sha256": backend.sha256(model),
                "engine_sha256": backend.sha256(engine), "tensorrt": "10.fake",
                "device": {"uuid": "test-device"}, "obs_dim": 49,
                "precision": "float32_no_tf32"}
        Path(str(engine)+".json").write_text(json.dumps(meta))
        context = FakeContext()
        engine_obj = SimpleNamespace(num_io_tensors=2, get_tensor_name=lambda i: ["obs", "actions"][i],
            get_tensor_dtype=lambda name: 0, get_tensor_location=lambda name: 1,
            get_tensor_format=lambda name: 0, get_tensor_mode=lambda name: int(name != "obs"),
            create_execution_context=lambda: context)
        logger = lambda level: None
        logger.WARNING = 1
        trt = SimpleNamespace(__version__="10.fake", float32=0, Logger=logger,
            TensorLocation=SimpleNamespace(DEVICE=1), TensorFormat=SimpleNamespace(LINEAR=0),
            TensorIOMode=SimpleNamespace(INPUT=0),
            Runtime=lambda log: SimpleNamespace(deserialize_cuda_engine=lambda data: engine_obj))
        with patch.dict(sys.modules, {"tensorrt": trt}), patch.object(backend, "CudaRuntime", FakeCuda):
            session = backend.TensorRTSession(model, engine)
        return session, model, engine, trt

    def test_device_output_is_synchronized_and_independent_host_copy(self):
        with tempfile.TemporaryDirectory() as directory:
            session, _, _, _ = self.session(directory)
            observation = np.arange(49, dtype=np.float32).reshape(1, 49)
            first = session.run(["actions"], {"obs": observation})[0]
            np.testing.assert_array_equal(first, observation[:, :12] * 2)
            session.run(["actions"], {"obs": np.zeros((1, 49), np.float32)})
            np.testing.assert_array_equal(first, observation[:, :12] * 2)
            self.assertEqual(session.context.executions, 2)
            self.assertEqual(session.cuda.calls.count("cudaStreamSynchronize"), 2)
            session.close()
            session.close()
            with self.assertRaisesRegex(RuntimeError, "closed"):
                session.run(["actions"], {"obs": observation})

    def test_invalid_input_is_rejected_before_execution(self):
        with tempfile.TemporaryDirectory() as directory:
            session, _, _, _ = self.session(directory)
            for value in (np.zeros((1,49), np.float64), np.zeros((49,), np.float32),
                          np.full((1,49), np.nan, np.float32)):
                with self.assertRaises(ValueError):
                    session.run(["actions"], {"obs": value})
            self.assertEqual(session.context.executions, 0)
            session.close()

    def test_model_hash_mismatch_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            session, model, engine, trt = self.session(directory)
            session.close()
            model.write_bytes(b"different-model")
            with patch.dict(sys.modules, {"tensorrt": trt}), patch.object(backend, "CudaRuntime", FakeCuda):
                with self.assertRaisesRegex(ValueError, "model_sha256"):
                    backend.TensorRTSession(model, engine)

    def test_cpu_default_and_trt_raw_action_keep_policy_contract(self):
        session = SimpleNamespace(get_inputs=lambda: [SimpleNamespace(name="obs", type="tensor(float)", shape=[1,49])],
                                  get_outputs=lambda: [SimpleNamespace(name="action")],
                                  run=lambda *args: [np.arange(12, dtype=np.float32).reshape(1,12)])
        with patch("policy_runner.ort.InferenceSession", return_value=session) as cpu:
            policy = HumanoidPolicy("model.onnx")
            cpu.assert_called_once_with("model.onnx", providers=["CPUExecutionProvider"])
        with patch.object(backend, "TensorRTSession", return_value=session) as gpu:
            policy_gpu = HumanoidPolicy("model.onnx", inference_backend="tensorrt", engine_path="model.engine")
            gpu.assert_called_once_with("model.onnx", "model.engine", 49)
        obs_args = dict(accel_m_s2=np.zeros(3), gyro_rad_s=np.zeros(3), projected_gravity=np.array([0,0,-1]),
                        velocity_command=np.array([.2,0,0]), joint_position_policy=np.zeros(12), joint_velocity_policy=np.zeros(12))
        cpu_result, gpu_result = policy.step(**obs_args), policy_gpu.step(**obs_args)
        for lhs, rhs in zip(cpu_result[:3], gpu_result[:3]):
            np.testing.assert_array_equal(lhs, rhs)
        np.testing.assert_array_equal(policy_gpu.last_action, np.arange(12, dtype=np.float32))

    def test_onefoot_backend_transmits_width_and_preserves_left_mirror(self):
        session = SimpleNamespace(get_inputs=lambda: [SimpleNamespace(name="obs", type="tensor(float)", shape=[1,46])],
                                  get_outputs=lambda: [SimpleNamespace(name="action")],
                                  run=lambda *args: [np.arange(12, dtype=np.float32).reshape(1,12)])
        with patch.object(backend, "TensorRTSession", return_value=session) as gpu:
            policy = OneFootPolicy("model.onnx", "left", inference_backend="tensorrt", engine_path="one.engine")
            gpu.assert_called_once_with("model.onnx", "one.engine", 46)
        import config
        action = np.arange(12, dtype=np.float32)
        np.testing.assert_array_equal(policy.action_to_target(action),
            config.Q_DEFAULT + config.ACTION_SCALE * -np.concatenate((action[6:],action[:6])))

    def test_benchmark_requires_real_observation_schema_and_preserves_order(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/"obs.csv"
            columns = observation_columns(49)
            with path.open("w", newline="") as handle:
                writer = csv.writer(handle)
                writer.writerow(["step"] + columns)
                writer.writerow([123] + list(range(49)))
            np.testing.assert_array_equal(load_observations(path,49), np.arange(49,dtype=np.float32).reshape(1,49))
            with self.assertRaises(ValueError):
                load_observations(path,46)
            path.write_text("target_R_ankle,actual_R_ankle\n0,0\n")
            with self.assertRaisesRegex(ValueError,"schema"):
                load_observations(path,49)

    def test_synthetic_smoke_is_deterministic_and_uses_exact_schema(self):
        for width in (49,46):
            first = synthetic_smoke_observations(width)
            second = synthetic_smoke_observations(width)
            np.testing.assert_array_equal(first,second)
            self.assertEqual(first.shape,(256,width))
            self.assertEqual(first.dtype,np.float32)
            self.assertTrue(np.isfinite(first).all())
            np.testing.assert_array_equal(first[0],np.zeros(width,np.float32))
            if width == 49:
                self.assertTrue(np.all(first[first[:,9] == 0,11] == 0))
                self.assertTrue(np.all(first[:,12] == 0))
            else:
                self.assertTrue(set(first[:,9]).issubset({0.,1.}))


if __name__ == "__main__":
    unittest.main()
