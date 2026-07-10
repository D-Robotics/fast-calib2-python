# Contributing

## Data policy

Do not commit raw sensor captures, rosbag files, PCD files, camera images, calibration outputs, or sensor-specific calibration files. The repository intentionally contains only source code and a zero-valued configuration template.

## Before opening a pull request

1. Run `python scripts/calibrate_lidar_camera.py --help`.
2. Confirm `git status --ignored` does not show data tracked by Git.
3. Keep target dimensions and sensor intrinsics in a local configuration file, not in the example template.
4. Explain any algorithmic changes and include a reproducible command using placeholder paths only.
