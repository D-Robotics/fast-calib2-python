from pathlib import Path

import pytest

from fast_calib2.config import load_config


def test_load_config_rejects_zero_focal_length(tmp_path: Path):
    config = tmp_path / "calibration.yaml"
    config.write_text("camera:\n  fx: 0\n  fy: 1\n", encoding="utf-8")

    with pytest.raises(ValueError, match="non-zero"):
        load_config(config)


def test_load_config_returns_typed_configuration(tmp_path: Path):
    config = tmp_path / "calibration.yaml"
    config.write_text("camera:\n  fx: 1000\n  fy: 1001\n", encoding="utf-8")

    result = load_config(config)

    assert result.camera.fx == 1000
    assert result.lidar.topic == "/livox/lidar"