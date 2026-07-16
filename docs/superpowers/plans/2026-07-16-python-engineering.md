# FAST-Calib2 Python Engineering Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `subagent-driven-development` or `executing-plans` to implement this plan task-by-task.

**Goal:** Package the reference calibration workflow while preserving CLI behavior and calibration outputs.

**Architecture:** Move stable components into `fast_calib2`, retain detector heuristics in a pipeline module, and expose installed and legacy CLIs.

**Tech Stack:** Python 3.9+, NumPy, OpenCV ArUco, PyYAML, pytest, GitHub Actions.

- [x] Add typed configuration and validation.
- [x] Extract deterministic geometry and binary PCD I/O.
- [x] Package existing pipeline and compatibility entry point.
- [x] Add pytest suite, packaging metadata, and CI.
- [ ] Verify, commit, and push.