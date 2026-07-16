"""Configuration loading and validation for calibration runs."""

from pathlib import Path
from typing import Any, Dict

import yaml

from .models import BoardConfig, CalibrationConfig, CameraConfig, LidarConfig

DEFAULT_CONFIG: Dict[str, Dict[str, Any]] = {
    "camera": {"fx": 0.0, "fy": 0.0, "cx": 0.0, "cy": 0.0, "distortion": [0.0] * 5},
    "board": {
        "marker_size": 0.20, "delta_width_qr_center": 0.55, "delta_height_qr_center": 0.35,
        "delta_width_circles": 0.50, "delta_height_circles": 0.40, "circle_radius": 0.12,
        "annulus_half_width": 0.025, "marker_ids": [1, 2, 4, 3], "aruco_dictionary": "DICT_6X6_250",
    },
    "lidar": {
        "topic": "/livox/lidar", "x_min": 0.8, "x_max": 4.5, "intensity_threshold": 80.0,
        "cluster_tolerance": 0.20, "cluster_min_size": 30, "plane_ransac_threshold": 0.03,
        "plane_near_threshold": 0.04, "grid_step": 0.02, "rectangle_tolerance": 0.08,
    },
}


def load_config(path: Path) -> CalibrationConfig:
    """Load a YAML configuration and reject unknown or unsafe values."""
    with Path(path).open("r", encoding="utf-8-sig") as config_file:
        supplied = yaml.safe_load(config_file) or {}
    if not isinstance(supplied, dict):
        raise ValueError("The calibration configuration must be a mapping")

    values = {section: defaults.copy() for section, defaults in DEFAULT_CONFIG.items()}
    for section, section_values in supplied.items():
        if section not in values:
            raise ValueError(f"Unsupported config section: {section}")
        if not isinstance(section_values, dict):
            raise ValueError(f"Config section '{section}' must be a mapping")
        unknown = set(section_values) - set(values[section])
        if unknown:
            raise ValueError(f"Unsupported keys in '{section}': {sorted(unknown)}")
        values[section].update(section_values)

    camera = values["camera"]
    board = values["board"]
    lidar = values["lidar"]
    if float(camera["fx"]) == 0.0 or float(camera["fy"]) == 0.0:
        raise ValueError("Set non-zero camera.fx and camera.fy in your configuration file")
    if len(camera["distortion"]) != 5:
        raise ValueError("camera.distortion must contain [k1, k2, p1, p2, k3]")
    if len(board["marker_ids"]) != 4:
        raise ValueError("board.marker_ids must contain exactly four marker IDs")
    if float(lidar["x_min"]) >= float(lidar["x_max"]):
        raise ValueError("lidar.x_min must be less than lidar.x_max")

    return CalibrationConfig(
        camera=CameraConfig(float(camera["fx"]), float(camera["fy"]), float(camera["cx"]), float(camera["cy"]), tuple(map(float, camera["distortion"]))),
        board=BoardConfig(float(board["marker_size"]), float(board["delta_width_qr_center"]), float(board["delta_height_qr_center"]), float(board["delta_width_circles"]), float(board["delta_height_circles"]), float(board["circle_radius"]), float(board["annulus_half_width"]), tuple(map(int, board["marker_ids"])), str(board["aruco_dictionary"])),
        lidar=LidarConfig(str(lidar["topic"]), float(lidar["x_min"]), float(lidar["x_max"]), float(lidar["intensity_threshold"]), float(lidar["cluster_tolerance"]), int(lidar["cluster_min_size"]), float(lidar["plane_ransac_threshold"]), float(lidar["plane_near_threshold"]), float(lidar["grid_step"]), float(lidar["rectangle_tolerance"])),
    )