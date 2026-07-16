"""Typed value objects used by FAST-Calib2 Python."""

from dataclasses import dataclass
from typing import Tuple


@dataclass(frozen=True)
class CameraConfig:
    fx: float
    fy: float
    cx: float
    cy: float
    distortion: Tuple[float, float, float, float, float]


@dataclass(frozen=True)
class BoardConfig:
    marker_size: float
    delta_width_qr_center: float
    delta_height_qr_center: float
    delta_width_circles: float
    delta_height_circles: float
    circle_radius: float
    annulus_half_width: float
    marker_ids: Tuple[int, int, int, int]
    aruco_dictionary: str


@dataclass(frozen=True)
class LidarConfig:
    topic: str
    x_min: float
    x_max: float
    intensity_threshold: float
    cluster_tolerance: float
    cluster_min_size: int
    plane_ransac_threshold: float
    plane_near_threshold: float
    grid_step: float
    rectangle_tolerance: float


@dataclass(frozen=True)
class CalibrationConfig:
    camera: CameraConfig
    board: BoardConfig
    lidar: LidarConfig