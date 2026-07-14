# FAST-Calib2 Python

A Python reference implementation of the [FAST-Calib2][https://github.com/hku-mars/FAST-Calib2] workflow for LiDAR-camera extrinsic calibration with a reflective annular target.

It estimates `T_cam_lidar` by:

1. detecting the four ArUco markers and deriving four target centers in the camera frame;
2. extracting four reflective annulus centers from the LiDAR point cloud; and
3. solving the LiDAR-to-camera rigid transform with SVD, optionally across multiple scenes.

The repository follows the original FAST-Calib2 layout: `config/` holds parameters, `scripts/` holds executable tooling, and all acquisition data and generated outputs remain local.

> **Data policy:** No sensor captures, calibration images, bag files, PCD files, sensor intrinsics, or calibration results are included. Use the zero-valued template as the starting point for your own setup.

## Repository layout

```text
config/
  calibration.example.yaml  # public template; no real intrinsics
docs/                       # additional documentation
scripts/
  calibrate_lidar_camera.py # calibration entry point
CONTRIBUTING.md
LICENSE
requirements.txt
```

## Requirements

- Python 3.9+
- ROS1 Python environment **only** when using rosbag input (`rosbag`, `sensor_msgs`)
- OpenCV with the `aruco` module

Install Python dependencies:

```bash
python3 -m pip install -r requirements.txt
```

For bag-based point-cloud input, source your ROS1 environment before invoking the script:

```bash
source /opt/ros/noetic/setup.bash
```

## Configuration

Create a private configuration file from the template:

```bash
cp config/calibration.example.yaml config/calibration.yaml
```

Set your own camera intrinsics, distortion coefficients, LiDAR topic, target geometry, and LiDAR extraction thresholds. `config/calibration.yaml` is ignored by Git on purpose.

The supplied board geometry defaults match the original FAST-Calib2 reflective-annulus target. Change them if your physical target differs.

## Data layout

The bag-based workflow expects a directory of scenes:

```text
/path/to/scenes/
  scene_001/
    scene.bmp
    scene.bag
  scene_002/
    scene.bmp
    scene.bag
  scene_003/
    scene.bmp
    scene.bag
```

Each bag must contain a `sensor_msgs/PointCloud2` message with `x`, `y`, `z`, and `intensity` fields on the configured LiDAR topic. Scene directories and all raw data stay outside this repository.

## Run calibration

```bash
python3 scripts/calibrate_lidar_camera.py \
  --config config/calibration.yaml \
  --scenes /path/to/scenes \
  --out output
```

The command writes per-scene transforms and a joint `multi_calib_result.txt` to `--out`. Generated results are ignored by Git.

To jointly solve from three specific successful scene names:

```bash
python3 scripts/calibrate_lidar_camera.py \
  --config config/calibration.yaml \
  --scenes /path/to/scenes \
  --out output \
  --multi scene_001 scene_002 scene_003
```

## Notes and limitations

- This is a reference Python implementation tailored to reflective annulus targets and sparse LiDAR data; tune thresholds for your sensor and acquisition setup.
- Successful calibration requires diverse, static target poses and independent projection checks. A low training-scene RMSE alone does not establish absolute calibration accuracy.
- The multi-frame PCD mode remains available for non-repetitive solid-state LiDAR scans. It expects the local raw-data layout documented in the script and is intentionally not accompanied by sample captures.
- The original ROS/C++ implementation is available from HKU-MARS at the link above.

## License

This project is distributed under the GPL-2.0 license. See `LICENSE`.
