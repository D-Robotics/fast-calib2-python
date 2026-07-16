import numpy as np

from fast_calib2.geometry import sort_centers, svd_rigid


def test_svd_rigid_recovers_known_transform():
    source = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 3.0]])
    rotation = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    translation = np.array([2.0, -3.0, 4.0])
    destination = source @ rotation.T + translation

    transform, rmse = svd_rigid(source, destination)

    np.testing.assert_allclose(transform[:3, :3], rotation, atol=1e-12)
    np.testing.assert_allclose(transform[:3, 3], translation, atol=1e-12)
    assert rmse < 1e-12


def test_sort_centers_returns_four_unique_centers():
    centers = np.array([[1.0, 1.0, 0.0], [-1.0, 1.0, 0.0], [-1.0, -1.0, 0.0], [1.0, -1.0, 0.0]])

    ordered = sort_centers(centers, "camera")

    assert ordered.shape == (4, 3)
    assert len({tuple(point) for point in ordered}) == 4