# FAST-Calib2 Python Engineering Design

## Goal

Turn the existing offline reference script into an installable, testable Python package without changing its calibration pipeline or supported input layouts.

## Decision

Use incremental package extraction: typed YAML configuration, deterministic geometry, and binary PCD input become independently testable modules. The original detector heuristics remain in `fast_calib2.pipeline`, protecting field behavior while adding a packaged CLI and compatibility wrapper.

## Compatibility

- Python 3.9+ and ROS1 bag-mode remain supported.
- Existing YAML keys, `scene_*` layout, and result matrix names remain unchanged.
- The existing OpenCV 4.7+ ArUco and mixed-field binary PCD fixes are retained.

## Verification

Validate configuration, PCD decoding, rigid transforms, CLI help, syntax compilation, package build, and pytest in CI.