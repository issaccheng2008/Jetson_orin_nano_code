"""Offline benchmark contracts; the fake backend is not evidence of GPU execution."""
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import benchmark_vision_devices as benchmark


class ReferenceCudaDouble(benchmark.cuda_vision.CpuVisionBackend):
    device = "cuda"

    def __init__(self):
        super().__init__(requested="cuda")
        self.calls = 0

    def morphology(self, image, operations):
        self.calls += 1
        return super().morphology(image, operations)

    def diagnostics(self):
        result = super().diagnostics()
        result.update(vision_device_actual="cuda", vision_cuda_calls_total=self.calls,
                      vision_cuda_retained_cpu="test double: all operations execute on CPU")
        return result


class BenchmarkTests(unittest.TestCase):
    def test_value_and_foreground_mismatches_are_distinct(self):
        difference = benchmark.binary_difference(np.array([[0, 1, 255]], np.uint8),
                                                np.array([[0, 255, 0]], np.uint8))
        self.assertEqual(difference["mismatched_pixels"], 2)
        self.assertAlmostEqual(difference["foreground_mismatch_fraction"], 1 / 3)
        self.assertFalse(benchmark.binary_difference(np.zeros((2, 2)), np.zeros((3, 2)))["available"])

    def test_fields_detect_geometry_and_class_changes(self):
        changes = benchmark.field_changes({"shape": "star", "cy": .5, "quad": [[1, 2]]},
                                          {"shape": "circle", "cy": .5 + 1e-8, "quad": [[1, 3]]})
        self.assertEqual(set(changes), {"shape", "quad"})

    def test_images_and_even_video_samples_are_bounded(self):
        with tempfile.TemporaryDirectory() as directory:
            image = np.full((64, 80, 3), 190, np.uint8)
            cv2.imwrite(str(Path(directory) / "still.png"), image)
            writer = cv2.VideoWriter(str(Path(directory) / "video.avi"),
                                    cv2.VideoWriter_fourcc(*"MJPG"), 10., (80, 64))
            self.assertTrue(writer.isOpened())
            for index in range(10):
                writer.write(image - index)
            writer.release()
            samples = benchmark.load_samples(directory, 4)
            self.assertEqual(len(samples), 4)
            self.assertEqual([sample.frame for sample in samples if sample.frame is not None], [0, 4, 9])
            self.assertEqual(samples[-1].timestamp_s, .9)

    def test_identical_replay_state_and_timings_with_reference_double(self):
        bgr = np.full((180, 320, 3), 180, np.uint8)
        cv2.rectangle(bgr, (115, 90), (205, 165), (20, 20, 20), 3)
        cv2.circle(bgr, (160, 128), 16, (20, 20, 20), 3)
        samples = [benchmark.Sample("same-video.avi", frame, frame / 10, bgr.copy())
                   for frame in (0, 1, 2)]
        config = benchmark.camera_config.load()
        original_clock = benchmark.shape_module.time
        cpu_rows, cpu_arrays = benchmark.run_device(samples, benchmark.cuda_vision.CpuVisionBackend(),
                                                    config, repeats=2, warmup=1)
        fake_rows, fake_arrays = benchmark.run_device(samples, ReferenceCudaDouble(),
                                                      config, repeats=2, warmup=1)
        comparisons = benchmark.compare_runs(cpu_rows, cpu_arrays, fake_rows, fake_arrays)
        self.assertEqual(len(comparisons), 6)
        self.assertIs(benchmark.shape_module.time, original_clock)
        for comparison in comparisons:
            self.assertFalse(comparison["line_changes"])
            self.assertFalse(comparison["shape_changes"])
            self.assertTrue(all(value["mismatched_pixels"] == 0 for value in comparison["binary"].values()))
        self.assertGreater(sum(row["cuda_calls_delta"] for row in fake_rows), 0)
        summary = benchmark.timing_summary(cpu_rows)
        self.assertEqual(summary["line_total_ms"]["n"], 6)
        self.assertIn("shape_verification_ms", summary)
        self.assertIn("line_preprocess_ms", summary)

    def test_missing_cuda_persists_cpu_report_and_require_cuda_fails(self):
        with tempfile.TemporaryDirectory() as directory:
            inputs = Path(directory) / "inputs"
            inputs.mkdir()
            cv2.imwrite(str(inputs / "blank.png"), np.full((90, 160, 3), 180, np.uint8))
            outputs = Path(directory) / "outputs"
            real_factory = benchmark.cuda_vision.create_vision_backend

            def factory(device, **kwargs):
                if device == "cuda":
                    raise benchmark.cuda_vision.VisionCudaError("test unavailable CUDA")
                return real_factory(device, **kwargs)
            with patch.object(benchmark.cuda_vision, "create_vision_backend", side_effect=factory):
                code = benchmark.main(["--input-dir", str(inputs), "--output-dir", str(outputs),
                                       "--max-images", "1", "--repeats", "1", "--require-cuda"])
            report = json.loads((outputs / "report.json").read_text())
            self.assertEqual(code, 2)
            self.assertEqual(report["devices"]["cpu"]["status"], "ok")
            self.assertEqual(report["devices"]["cuda"]["status"], "error")
            self.assertEqual(report["comparison_summary"]["calls_compared"], 0)


if __name__ == "__main__":
    unittest.main()
