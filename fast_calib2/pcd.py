"""Minimal PCD reader used by the optional multi-frame calibration workflow.

Handles binary PCDs whose fields may include non-float32 columns (e.g. a Livox
capture with ``x y z intensity tag line offset_time`` where ``tag``/``line`` are
uint8/uint16). Only ``x``, ``y``, ``z`` and ``intensity`` are returned; every
other field is ignored. Field offsets are derived from the header SIZE/COUNT so
mixed-type records are read correctly.
"""

import numpy as np


def load_pcd(path):
    """Load binary PCD, returning ``(xyz, intensity)`` as float32 arrays.

    Raises ``ValueError`` if the file is not binary or lacks the required
    ``x``/``y``/``z``/``intensity`` fields (which must themselves be float32).
    """
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

    needed = ("x", "y", "z", "intensity")
    if any(f not in fields for f in needed):
        raise ValueError("PCD must provide x, y, z, and intensity fields")

    # Per-field byte offset within one point record (sum of preceding sizes*counts).
    offsets = {}
    offset = 0
    for name, size, count in zip(fields, sizes, counts):
        offsets[name] = offset
        offset += size * count
    record_size = offset
    if len(payload) % record_size:
        raise ValueError("PCD binary payload has an invalid length")

    # Validate the four required fields are float32 scalars; ignore the rest.
    for name in needed:
        idx = fields.index(name)
        if sizes[idx] != 4 or types[idx].upper() != "F" or counts[idx] != 1:
            raise ValueError(f"PCD field '{name}' must be a scalar float32 value")

    raw = np.frombuffer(payload, dtype=np.uint8).reshape(-1, record_size)
    n = raw.shape[0]
    xyz = np.empty((n, 3), dtype=np.float32)
    for col, name in enumerate(("x", "y", "z")):
        off = offsets[name]
        xyz[:, col] = np.frombuffer(raw[:, off:off + 4].tobytes(), dtype="<f4")
    intensity = np.frombuffer(raw[:, offsets["intensity"]:offsets["intensity"] + 4].tobytes(), dtype="<f4")
    return xyz, intensity
