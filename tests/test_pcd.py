import struct
from pathlib import Path

import numpy as np

from fast_calib2.pcd import load_pcd


def test_load_pcd_reads_mixed_field_binary_records(tmp_path: Path):
    pcd = tmp_path / "cloud.pcd"
    header = b"""# .PCD v0.7\nVERSION 0.7\nFIELDS x y z intensity tag line\nSIZE 4 4 4 4 1 2\nTYPE F F F F U U\nCOUNT 1 1 1 1 1 1\nWIDTH 2\nHEIGHT 1\nPOINTS 2\nDATA binary\n"""
    points = struct.pack("<ffffBHffffBH", 1, 2, 3, 4, 9, 10, 5, 6, 7, 8, 11, 12)
    pcd.write_bytes(header + points)

    xyz, intensity = load_pcd(pcd)

    np.testing.assert_allclose(xyz, [[1, 2, 3], [5, 6, 7]])
    np.testing.assert_allclose(intensity, [4, 8])