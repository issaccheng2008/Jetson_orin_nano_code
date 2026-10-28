"""Offline ONNX -> FP32 TensorRT 10 engine build. Does not import motor code."""
import argparse
import json
from tensorrt_backend import build_engine


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--engine", required=True)
    parser.add_argument("--obs-dim", type=int, choices=(49, 46), default=49)
    parser.add_argument("--workspace-mib", type=int, default=256)
    args = parser.parse_args()
    if args.workspace_mib <= 0:
        parser.error("--workspace-mib must be positive")
    print(json.dumps(build_engine(args.model, args.engine, args.obs_dim, args.workspace_mib), indent=2))


if __name__ == "__main__":
    main()
