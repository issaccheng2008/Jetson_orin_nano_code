#!/usr/bin/env python3
from __future__ import annotations

import argparse
import time

import numpy as np
import onnxruntime as ort


parser = argparse.ArgumentParser()
parser.add_argument("model")
parser.add_argument("--runs", type=int, default=1000)
args = parser.parse_args()

session = ort.InferenceSession(args.model, providers=["CPUExecutionProvider"])
print("Providers:", session.get_providers())
for value in session.get_inputs():
    print("Input:", value.name, value.shape, value.type)
for value in session.get_outputs():
    print("Output:", value.name, value.shape, value.type)

input_name = session.get_inputs()[0].name
obs = np.zeros((1, 48), dtype=np.float32)
for _ in range(20):
    session.run(None, {input_name: obs})

start = time.perf_counter()
for _ in range(args.runs):
    output = session.run(None, {input_name: obs})[0]
elapsed = time.perf_counter() - start
print("Output shape:", output.shape)
print("Finite:", np.isfinite(output).all())
print(f"Mean inference time: {elapsed / args.runs * 1000.0:.4f} ms")
