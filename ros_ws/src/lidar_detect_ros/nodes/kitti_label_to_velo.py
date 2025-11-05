#!/usr/bin/env python3
import numpy as np
import sys, os

def _reshape_or_none(arr, rows, cols):
    if arr is None:
        return None
    a = np.asarray(arr, dtype=float).ravel()
    if a.size != rows * cols:
        return None
    return a.reshape(rows, cols)

def read_calib(calib_file):
    """
    Parse KITTI calib file. We only need:
      - R0_rect (3x3)   (sometimes called R_rect_00)
      - Tr_velo_to_cam (3x4) (sometimes Tr_velo_to_cam0)
    Returns (R0_rect_3x3, Tr_velo_to_cam_3x4) or raises ValueError with details.
    """
    data = {}
    with open(calib_file, "r") as f:
        for line in f:
            line = line.strip()
            if not line or ":" not in line:
                continue
            key, values = line.split(":", 1)
            try:
                nums = [float(x) for x in values.strip().split()]
                data[key.strip()] = np.array(nums, dtype=float)
            except Exception:
                # Ignore non-numeric lines
                continue

    # Get R0 (may be under different key)
    R0 = data.get("R0_rect", None)
    if R0 is None:
        R0 = data.get("R_rect_00", None)

    # Get Tr (may be under different key)
    Tr = data.get("Tr_velo_to_cam", None)
    if Tr is None:
        Tr = data.get("Tr_velo_to_cam0", None)

    R0 = _reshape_or_none(R0, 3, 3)
    Tr = _reshape_or_none(Tr, 3, 4)

    missing = []
    if R0 is None:
        missing.append("R0_rect (3x3) / R_rect_00")
    if Tr is None:
        missing.append("Tr_velo_to_cam (3x4) / Tr_velo_to_cam0")
    if missing:
        keys = ", ".join(sorted(data.keys()))
        raise ValueError(f"Missing or malformed calib keys: {', '.join(missing)}. "
                         f"Found keys: {keys}")

    return R0, Tr

def cam_to_velo(cam_loc_xyz, Tr, R0):
    """
    Convert camera rectified coords (3,) -> Velodyne coords (3,).
    X_velo = inv(Tr_velo_to_cam) * inv(R0_rect) * X_cam
    """
    # Build 4x4 matrices
    R0_4 = np.eye(4, dtype=float)
    R0_4[:3, :3] = R0

    Tr_4 = np.eye(4, dtype=float)
    Tr_4[:3, :4] = Tr

    X_cam = np.array([cam_loc_xyz[0], cam_loc_xyz[1], cam_loc_xyz[2], 1.0], dtype=float)

    try:
        V_from_C = np.linalg.inv(Tr_4) @ np.linalg.inv(R0_4)
    except np.linalg.LinAlgError as e:
        raise ValueError(f"Calibration matrices not invertible: {e}")

    X_velo = V_from_C @ X_cam
    return X_velo[:3]

def read_labels(label_file):
    objs = []
    with open(label_file, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 15:
                continue
            cls = parts[0]
            if cls.lower() == "dontcare":
                continue
            try:
                # dims = h,w,l (parts[8:11]) not needed here, but you keep if useful later
                loc = [float(parts[11]), float(parts[12]), float(parts[13])]  # camera coords
                objs.append({"type": cls, "loc_cam": loc})
            except Exception:
                # skip malformed line
                continue
    return objs

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: rosrun my_lidar_processing kitti_label_to_velo.py <calib.txt> <label.txt>")
        sys.exit(1)

    calib_file = sys.argv[1]
    label_file = sys.argv[2]

    if not (os.path.isfile(calib_file) and os.path.isfile(label_file)):
        print("[ERROR] File not found (calib or label).")
        sys.exit(1)

    try:
        R0, Tr = read_calib(calib_file)
    except Exception as e:
        # Print a clear, single-line error (your matcher shows this stderr)
        print(f"[ERROR] Calib parse failed: {e}")
        sys.exit(1)

    labels = read_labels(label_file)
    if not labels:
        # Still succeed with no lines printed (matcher handles empty)
        sys.exit(0)

    for obj in labels:
        xyz_velo = cam_to_velo(obj["loc_cam"], Tr, R0)
        # Keep format your matcher already parses (last 3 floats are coords):
        print(f"{obj['type']} at velodyne coords: ({xyz_velo[0]:.2f}, {xyz_velo[1]:.2f}, {xyz_velo[2]:.2f})")
