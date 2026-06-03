#!/usr/bin/env python3
import json
from pathlib import Path

import numpy as np
import yaml


def quat_xyzw_to_matrix(q):
    x, y, z, w = q
    n = x*x + y*y + z*z + w*w
    if n < 1e-12:
        return np.eye(3)

    s = 2.0 / n
    xx, yy, zz = x*x*s, y*y*s, z*z*s
    xy, xz, yz = x*y*s, x*z*s, y*z*s
    wx, wy, wz = w*x*s, w*y*s, w*z*s

    return np.array([
        [1.0 - (yy + zz), xy - wz, xz + wy],
        [xy + wz, 1.0 - (xx + zz), yz - wx],
        [xz - wy, yz + wx, 1.0 - (xx + yy)],
    ], dtype=float)


def matrix_to_quat_xyzw(R):
    m = np.asarray(R, dtype=float)
    trace = np.trace(m)

    if trace > 0.0:
        s = np.sqrt(trace + 1.0) * 2.0
        w = 0.25 * s
        x = (m[2, 1] - m[1, 2]) / s
        y = (m[0, 2] - m[2, 0]) / s
        z = (m[1, 0] - m[0, 1]) / s
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2.0
        w = (m[2, 1] - m[1, 2]) / s
        x = 0.25 * s
        y = (m[0, 1] + m[1, 0]) / s
        z = (m[0, 2] + m[2, 0]) / s
    elif m[1, 1] > m[2, 2]:
        s = np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2.0
        w = (m[0, 2] - m[2, 0]) / s
        x = (m[0, 1] + m[1, 0]) / s
        y = 0.25 * s
        z = (m[1, 2] + m[2, 1]) / s
    else:
        s = np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2.0
        w = (m[1, 0] - m[0, 1]) / s
        x = (m[0, 2] + m[2, 0]) / s
        y = (m[1, 2] + m[2, 1]) / s
        z = 0.25 * s

    q = np.array([x, y, z, w], dtype=float)
    q = q / np.linalg.norm(q)
    return q


def make_transform(translation, quaternion_xyzw):
    T = np.eye(4, dtype=float)
    T[:3, :3] = quat_xyzw_to_matrix(quaternion_xyzw)
    T[:3, 3] = np.asarray(translation, dtype=float)
    return T


def make_transform_from_rotation_matrix(translation, rotation_matrix):
    T = np.eye(4, dtype=float)
    T[:3, :3] = np.asarray(rotation_matrix, dtype=float)
    T[:3, 3] = np.asarray(translation, dtype=float)
    return T


def main():
    pkg_dir = Path.home() / "franka_grasp_ws/src/franka_grasp_demo"

    calib_yaml = pkg_dir / "config/external_camera_calibration.yaml"
    aruco_json = pkg_dir / "data/external_aruco_calib_8cm/aruco_pose_camera.json"
    output_yaml = pkg_dir / "config/camera_extrinsic.yaml"

    with open(calib_yaml, "r") as f:
        cfg = yaml.safe_load(f)

    with open(aruco_json, "r") as f:
        aruco = json.load(f)

    T_base_marker = make_transform(
        cfg["T_base_marker"]["translation"],
        cfg["T_base_marker"]["quaternion_xyzw"],
    )

    T_camera_marker = make_transform_from_rotation_matrix(
        aruco["T_camera_marker"]["translation"],
        aruco["T_camera_marker"]["rotation_matrix"],
    )

    # T_base_camera = T_base_marker * inverse(T_camera_marker)
    T_base_camera = T_base_marker @ np.linalg.inv(T_camera_marker)

    t = T_base_camera[:3, 3]
    q = matrix_to_quat_xyzw(T_base_camera[:3, :3])

    result = {
        "base_frame": cfg.get("base_frame", "base"),
        "camera_frame": cfg.get("camera_frame", "camera_color_optical_frame"),
        "T_base_camera": {
            "translation": [float(t[0]), float(t[1]), float(t[2])],
            "quaternion_xyzw": [float(q[0]), float(q[1]), float(q[2]), float(q[3])],
            "rotation_matrix": T_base_camera[:3, :3].tolist(),
        },
        "T_grasp_tcp": {
            "translation": [0.0, 0.0, 0.0],
            "quaternion_xyzw": [0.0, 0.0, 0.0, 1.0],
        },
        "calibration_source": {
            "marker_size": cfg.get("marker_size", 0.08),
            "aruco_pose_camera_json": str(aruco_json),
            "external_camera_calibration_yaml": str(calib_yaml),
        },
        "note": "External fixed camera calibration. T_base_camera maps camera_color_optical_frame into base."
    }

    print("Computed T_base_camera:")
    print(yaml.safe_dump(result, sort_keys=False))

    output_yaml.parent.mkdir(parents=True, exist_ok=True)
    with open(output_yaml, "w") as f:
        yaml.safe_dump(result, f, sort_keys=False)

    print(f"Saved: {output_yaml}")


if __name__ == "__main__":
    main()
