#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import numpy as np
import yaml


def quat_xyzw_to_matrix(q):
    x, y, z, w = q
    n = x * x + y * y + z * z + w * w
    if n < 1e-12:
        return np.eye(3)

    s = 2.0 / n

    xx = x * x * s
    yy = y * y * s
    zz = z * z * s
    xy = x * y * s
    xz = x * z * s
    yz = y * z * s
    wx = w * x * s
    wy = w * y * s
    wz = w * z * s

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


def transform_from_grasp_json(grasp):
    T = np.eye(4, dtype=float)
    T[:3, 3] = np.asarray(grasp["position"], dtype=float)

    if "rotation_matrix" in grasp:
        T[:3, :3] = np.asarray(grasp["rotation_matrix"], dtype=float)
    else:
        T[:3, :3] = quat_xyzw_to_matrix(grasp["orientation"])

    return T


def load_yaml(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def load_json(path):
    with open(path, "r") as f:
        return json.load(f)


def save_json(obj, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)
    print(f"Saved: {path}")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input_json",
        default=str(Path.home() / "franka_grasp_ws/src/franka_grasp_demo/data/graspnet_result/grasp_result_camera_roi.json"),
    )
    parser.add_argument(
        "--output_json",
        default=str(Path.home() / "franka_grasp_ws/src/franka_grasp_demo/data/graspnet_result/grasp_result.json"),
    )
    parser.add_argument(
        "--extrinsic_yaml",
        default=str(Path.home() / "franka_grasp_ws/src/franka_grasp_demo/config/camera_extrinsic.yaml"),
    )

    # 仅用于 fake hardware 软件闭环测试：强行覆盖输出位置/姿态。
    # 真机不要用 override。
    parser.add_argument("--override_position_base", type=float, nargs=3, default=None)
    parser.add_argument("--override_orientation_xyzw", type=float, nargs=4, default=None)

    args = parser.parse_args()

    grasp_camera = load_json(args.input_json)
    cfg = load_yaml(args.extrinsic_yaml)

    base_frame = cfg.get("base_frame", "base")
    camera_frame = cfg.get("camera_frame", "camera_color_optical_frame")

    if grasp_camera.get("frame_id") != camera_frame:
        print(
            f"[WARN] input frame_id={grasp_camera.get('frame_id')} "
            f"but config camera_frame={camera_frame}"
        )

    T_base_camera = make_transform(
        cfg["T_base_camera"]["translation"],
        cfg["T_base_camera"]["quaternion_xyzw"],
    )

    T_grasp_tcp = make_transform(
        cfg["T_grasp_tcp"]["translation"],
        cfg["T_grasp_tcp"]["quaternion_xyzw"],
    )

    T_camera_grasp = transform_from_grasp_json(grasp_camera)

    # 核心转换：
    # T_base_tcp = T_base_camera * T_camera_grasp * T_grasp_tcp
    T_base_tcp = T_base_camera @ T_camera_grasp @ T_grasp_tcp

    position = T_base_tcp[:3, 3]
    orientation = matrix_to_quat_xyzw(T_base_tcp[:3, :3])

    if args.override_position_base is not None:
        print("[WARN] overriding output base position for fake hardware test.")
        position = np.asarray(args.override_position_base, dtype=float)

    if args.override_orientation_xyzw is not None:
        print("[WARN] overriding output base orientation for fake hardware test.")
        orientation = np.asarray(args.override_orientation_xyzw, dtype=float)
        orientation = orientation / np.linalg.norm(orientation)

    result = {
        "frame_id": base_frame,
        "position": [
            float(position[0]),
            float(position[1]),
            float(position[2]),
        ],
        "orientation": [
            float(orientation[0]),
            float(orientation[1]),
            float(orientation[2]),
            float(orientation[3]),
        ],
        "rotation_matrix": T_base_tcp[:3, :3].tolist(),
        "width": grasp_camera.get("width"),
        "score": grasp_camera.get("score"),
        "source_camera_grasp": grasp_camera,
        "note": (
            "This is transformed to base frame. "
            "If extrinsic is not calibrated, this is only for fake hardware testing."
        ),
    }

    print("Transformed grasp:")
    print(json.dumps(result, indent=2))

    save_json(result, args.output_json)


if __name__ == "__main__":
    main()
