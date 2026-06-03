#!/usr/bin/env python3
import argparse
import json
import math
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
    ], dtype=np.float64)


def matrix_to_quat_xyzw(R):
    m = np.asarray(R, dtype=np.float64)
    tr = np.trace(m)

    if tr > 0.0:
        s = math.sqrt(tr + 1.0) * 2.0
        w = 0.25 * s
        x = (m[2, 1] - m[1, 2]) / s
        y = (m[0, 2] - m[2, 0]) / s
        z = (m[1, 0] - m[0, 1]) / s
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = math.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2.0
        w = (m[2, 1] - m[1, 2]) / s
        x = 0.25 * s
        y = (m[0, 1] + m[1, 0]) / s
        z = (m[0, 2] + m[2, 0]) / s
    elif m[1, 1] > m[2, 2]:
        s = math.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2.0
        w = (m[0, 2] - m[2, 0]) / s
        x = (m[0, 1] + m[1, 0]) / s
        y = 0.25 * s
        z = (m[1, 2] + m[2, 1]) / s
    else:
        s = math.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2.0
        w = (m[1, 0] - m[0, 1]) / s
        x = (m[0, 2] + m[2, 0]) / s
        y = (m[1, 2] + m[2, 1]) / s
        z = 0.25 * s

    q = np.array([x, y, z, w], dtype=np.float64)
    q = q / np.linalg.norm(q)
    return q


def make_T(translation, quat_xyzw):
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = quat_xyzw_to_matrix(quat_xyzw)
    T[:3, 3] = np.asarray(translation, dtype=np.float64)
    return T


def rot_quat(axis, deg):
    a = math.radians(deg)
    s = math.sin(a / 2.0)
    c = math.cos(a / 2.0)

    if axis == "x":
        return [s, 0.0, 0.0, c]
    if axis == "y":
        return [0.0, s, 0.0, c]
    if axis == "z":
        return [0.0, 0.0, s, c]
    raise ValueError(axis)


def compensation_T(name):
    if name == "identity":
        q = [0.0, 0.0, 0.0, 1.0]
    elif name == "rz90":
        q = rot_quat("z", 90)
    elif name == "rz-90":
        q = rot_quat("z", -90)
    elif name == "rz180":
        q = rot_quat("z", 180)
    elif name == "rx90":
        q = rot_quat("x", 90)
    elif name == "rx-90":
        q = rot_quat("x", -90)
    elif name == "ry90":
        q = rot_quat("y", 90)
    elif name == "ry-90":
        q = rot_quat("y", -90)
    else:
        raise ValueError(f"unknown compensation: {name}")

    return make_T([0.0, 0.0, 0.0], q), q


def in_workspace(p):
    x, y, z = p
    return (
        0.25 <= x <= 0.75 and
        -0.35 <= y <= 0.35 and
        0.02 <= z <= 0.35
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_json", required=True)
    parser.add_argument("--camera_extrinsic", default="config/camera_extrinsic.yaml")
    parser.add_argument("--output_json", required=True)
    parser.add_argument(
        "--compensation",
        default="identity",
        choices=["identity", "rz90", "rz-90", "rz180", "rx90", "rx-90", "ry90", "ry-90"],
    )
    parser.add_argument("--only_target_region", action="store_true")
    args = parser.parse_args()

    input_path = Path(args.input_json)
    output_path = Path(args.output_json)
    extrinsic_path = Path(args.camera_extrinsic)

    data = json.load(open(input_path, "r"))
    cfg = yaml.safe_load(open(extrinsic_path, "r"))

    T_base_camera = make_T(
        cfg["T_base_camera"]["translation"],
        cfg["T_base_camera"]["quaternion_xyzw"],
    )

    T_grasp_tcp, q_comp = compensation_T(args.compensation)

    out = {
        "source_file": str(input_path),
        "frame_id": "base",
        "camera_frame": data.get("frame_id", "camera_color_optical_frame"),
        "roi": data.get("roi"),
        "num_candidates_input": len(data["grasps"]),
        "compensation": args.compensation,
        "T_grasp_tcp_quaternion_xyzw": q_comp,
        "grasps": [],
    }

    for g in data["grasps"]:
        if args.only_target_region and not g.get("in_target_region", False):
            continue

        T_camera_grasp = make_T(g["position"], g["orientation"])
        T_base_tcp = T_base_camera @ T_camera_grasp @ T_grasp_tcp

        pos = T_base_tcp[:3, 3]
        quat = matrix_to_quat_xyzw(T_base_tcp[:3, :3])

        new_g = {
            "id": int(g["id"]),
            "frame_id": "base",
            "position": [float(pos[0]), float(pos[1]), float(pos[2])],
            "orientation": [float(quat[0]), float(quat[1]), float(quat[2]), float(quat[3])],
            "rotation_matrix": T_base_tcp[:3, :3].tolist(),
            "width": float(g["width"]),
            "score": float(g["score"]),
            "pixel": g.get("pixel"),
            "in_target_region": bool(g.get("in_target_region", False)),
            "in_workspace": bool(in_workspace(pos)),
            "source_camera_grasp": g,
        }
        out["grasps"].append(new_g)

    out["num_candidates_output"] = len(out["grasps"])
    out["num_in_workspace"] = sum(1 for g in out["grasps"] if g["in_workspace"])
    out["num_in_target_region"] = sum(1 for g in out["grasps"] if g["in_target_region"])

    output_path.parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(output_path, "w"), indent=2)

    print(f"Saved: {output_path}")
    print(f"compensation: {args.compensation}")
    print(f"output candidates: {out['num_candidates_output']}")
    print(f"in target region: {out['num_in_target_region']}")
    print(f"in workspace: {out['num_in_workspace']}")

    print("\nTop 15 candidates:")
    sorted_grasps = sorted(out["grasps"], key=lambda x: x["score"], reverse=True)
    for g in sorted_grasps[:15]:
        p = g["position"]
        print(
            f"id={g['id']:3d} score={g['score']:.3f} "
            f"width={g['width']:.3f} "
            f"base=({p[0]:.3f}, {p[1]:.3f}, {p[2]:.3f}) "
            f"target={g['in_target_region']} workspace={g['in_workspace']}"
        )


if __name__ == "__main__":
    main()
