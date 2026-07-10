"""Minimal PCD reader used by the optional multi-frame calibration workflow."""

import numpy as np


def load_pcd(path):
    """Load binary PCD files containing float32 x, y, z, and intensity fields."""
    with open(path, "rb") as pcd_file:
        header_lines = []
        while True:
            line = pcd_file.readline()
            if not line:
                raise ValueError(f"Incomplete PCD header: {path}")
            header_lines.append(line.decode("ascii", errors="replace").strip())
            if header_lines[-1].upper().startswith("DATA"):
                break
        payload = pcd_file.read()

    header = {}
    for line in header_lines:
        key, *values = line.split()
        if key:
            header[key.upper()] = values
    if header.get("DATA", [""])[0].lower() != "binary":
        raise ValueError("Only binary PCD files are supported")

    fields = header.get("FIELDS", [])
    sizes = [int(value) for value in header.get("SIZE", [])]
    types = header.get("TYPE", [])
    counts = [int(value) for value in header.get("COUNT", ["1"] * len(fields))]
    if not fields or len(fields) != len(sizes) or len(fields) != len(types):
        raise ValueError("Invalid PCD field declaration")
    if any(field not in fields for field in ("x", "y", "z", "intensity")):
        raise ValueError("PCD must provide x, y, z, and intensity fields")
    if any(size != 4 or field_type.upper() != "F" or count != 1 for size, field_type, count in zip(sizes, types, counts)):
        raise ValueError("PCD fields must be scalar float32 values")

    dtype = np.dtype([(field, "<f4") for field in fields])
    if len(payload) % dtype.itemsize:
        raise ValueError("PCD binary payload has an invalid length")
    points = np.frombuffer(payload, dtype=dtype)
    xyz = np.column_stack([points["x"], points["y"], points["z"]]).astype(np.float32)
    intensity = points["intensity"].astype(np.float32)
    return xyz, intensity
