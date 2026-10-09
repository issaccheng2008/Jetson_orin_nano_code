# Offline recording analysis implementation plan

Approved scope: crop one grouped test at both valves passed, or button release in button mode; plot deviation angle plus visual/model WZ on one dual-axis figure, IMU on another. Preserve input files.

1. Add regression tests for asynchronous valves, button mode, missing start, shared Unix-time cropping and quaternion conversion. Run red.
2. Log gate mode and latched QR/shape/button states per visual frame without changing control decisions.
3. Add standalone standard-library CSV/JSON processing and optional offline matplotlib plotting. Discover exactly one visual and one control file per session, reject ambiguous selection. Export cropped CSVs, PNGs and summary in analysis/.
4. Document CLI, dependencies, timestamp semantics and legacy manual start override. Verify synthetic end-to-end output and inspect figures; run affected and full visual tests, review and push policy49flitter.
