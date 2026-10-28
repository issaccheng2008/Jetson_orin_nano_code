# Offline vision device benchmark

Warmup excluded; synchronous wall time includes transfers and rendering. Backend stages overlap detector stages; do not add them together. line_remainder includes geometry, photometry and rendering.

Samples: 5; repeats: 3; warmup rounds: 1.

| Device | Status | Stage | p50 ms | p95 ms |
|---|---|---|---:|---:|
| cpu | ok | line.backend.gray_ms | 0.110 | 0.329 |
| cpu | ok | line.backend.morphology_ms | 3.184 | 4.980 |
| cpu | ok | line.backend.warp_ms | 0.889 | 1.340 |
| cpu | ok | line_preprocess_ms | 6.595 | 10.959 |
| cpu | ok | line_remainder_ms | 104.402 | 113.891 |
| cpu | ok | line_total_ms | 113.157 | 120.691 |
| cpu | ok | shape.backend.gray_ms | 1.142 | 1.834 |
| cpu | ok | shape.backend.morphology_ms | 7.297 | 10.166 |
| cpu | ok | shape.backend.resize_ms | 0.433 | 1.258 |
| cpu | ok | shape_call_ms | 79.764 | 100.902 |
| cpu | ok | shape_classify_ms | 0.000 | 0.000 |
| cpu | ok | shape_preprocess_ms | 26.077 | 31.335 |
| cpu | ok | shape_presence_ms | 12.163 | 19.694 |
| cpu | ok | shape_proposals_ms | 37.610 | 51.559 |
| cpu | ok | shape_total_ms | 79.616 | 100.697 |
| cpu | ok | shape_verification_ms | 1.385 | 5.363 |
| cuda | error: OpenCV reports no usable CUDA device | — | — | — |

Comparison summary: `{"calls_compared": 0, "line_changed_calls": 0, "shape_changed_calls": 0, "classification_changed_calls": 0, "presence_changed_calls": 0, "geometry_changed_calls": 0, "binary_changed_calls": 0}`.

See report.json for every input pixel hash, source hash, backend retained CPU reasons and per-call changes. CPU-only/fallback measurements provide no CUDA speed evidence.
