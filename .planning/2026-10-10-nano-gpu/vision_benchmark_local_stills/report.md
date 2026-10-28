# Offline vision device benchmark

Warmup excluded; synchronous wall time includes transfers and rendering. Backend stages overlap detector stages; do not add them together. line_remainder includes geometry, photometry and rendering.

Samples: 6; repeats: 3; warmup rounds: 1.

| Device | Status | Stage | p50 ms | p95 ms |
|---|---|---|---:|---:|
| cpu | ok | line.backend.gray_ms | 0.124 | 0.510 |
| cpu | ok | line.backend.morphology_ms | 3.320 | 4.926 |
| cpu | ok | line.backend.warp_ms | 1.039 | 1.644 |
| cpu | ok | line_preprocess_ms | 7.863 | 11.914 |
| cpu | ok | line_remainder_ms | 119.352 | 132.981 |
| cpu | ok | line_total_ms | 129.612 | 146.913 |
| cpu | ok | shape.backend.gray_ms | 0.868 | 1.657 |
| cpu | ok | shape.backend.morphology_ms | 1.995 | 2.597 |
| cpu | ok | shape.backend.resize_ms | 0.296 | 0.835 |
| cpu | ok | shape_call_ms | 106.922 | 163.436 |
| cpu | ok | shape_classify_ms | 2.096 | 2.637 |
| cpu | ok | shape_preprocess_ms | 30.305 | 43.569 |
| cpu | ok | shape_presence_ms | 0.000 | 0.000 |
| cpu | ok | shape_proposals_ms | 64.489 | 105.783 |
| cpu | ok | shape_total_ms | 106.503 | 163.233 |
| cpu | ok | shape_verification_ms | 10.137 | 17.883 |
| cuda | error: OpenCV reports no usable CUDA device | — | — | — |

Comparison summary: `{"calls_compared": 0, "line_changed_calls": 0, "shape_changed_calls": 0, "classification_changed_calls": 0, "presence_changed_calls": 0, "geometry_changed_calls": 0, "binary_changed_calls": 0}`.

See report.json for every input pixel hash, source hash, backend retained CPU reasons and per-call changes. CPU-only/fallback measurements provide no CUDA speed evidence.
