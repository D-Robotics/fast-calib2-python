"""Deterministic geometric operations shared by calibration workflows."""

import numpy as np


def svd_rigid(source: np.ndarray, destination: np.ndarray):
    """Return the homogeneous transform mapping source points to destination points."""
    source = np.asarray(source, dtype=np.float64)
    destination = np.asarray(destination, dtype=np.float64)
    if source.shape != destination.shape or source.ndim != 2 or source.shape[1] != 3 or len(source) < 3:
        raise ValueError("source and destination must be matching Nx3 arrays with N >= 3")
    source_center = source.mean(axis=0)
    destination_center = destination.mean(axis=0)
    covariance = (source - source_center).T @ (destination - destination_center)
    left, _, right_t = np.linalg.svd(covariance)
    rotation = right_t.T @ left.T
    if np.linalg.det(rotation) < 0:
        right_t[-1] *= -1
        rotation = right_t.T @ left.T
    translation = destination_center - rotation @ source_center
    transform = np.eye(4)
    transform[:3, :3] = rotation
    transform[:3, 3] = translation
    rmse = float(np.sqrt(np.mean(np.sum((source @ rotation.T + translation - destination) ** 2, axis=1))))
    return transform, rmse


def sort_centers(centers: np.ndarray, mode: str = "camera") -> np.ndarray:
    """Return four target centers in a repeatable counter-clockwise order."""
    points = np.asarray(centers, dtype=np.float64)
    if points.shape != (4, 3):
        raise ValueError("centers must be a 4x3 array")
    origin = points.mean(axis=0)
    angles = np.arctan2(points[:, 1] - origin[1], points[:, 0] - origin[0])
    ordered = points[np.argsort(angles)]
    if mode not in {"camera", "lidar"}:
        raise ValueError("mode must be 'camera' or 'lidar'")
    return ordered