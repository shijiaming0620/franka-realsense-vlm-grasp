import argparse
import json
import yaml
from pathlib import Path

import cv2
import numpy as np


def load_camera_info(path):
    info = json.load(open(path))
    K_raw = info.get("k") or info.get("K")
    if K_raw is None:
        raise RuntimeError("camera_info.json missing K/k")
    return np.array(K_raw, dtype=float).reshape(3, 3)


def depth_at(depth, u, v, radius=10):
    h, w = depth.shape[:2]
    x1, x2 = max(0, u - radius), min(w, u + radius + 1)
    y1, y2 = max(0, v - radius), min(h, v + radius + 1)

    patch = depth[y1:y2, x1:x2]
    valid = patch[patch > 0]

    if valid.size == 0:
        return None

    d = float(np.median(valid))

    if depth.dtype == np.uint16:
        d = d / 1000.0

    return d


def camera_point_from_pixel(u, v, z, K):
    fx = K[0, 0]
    fy = K[1, 1]
    cx = K[0, 2]
    cy = K[1, 2]

    x = (u - cx) * z / fx
    y = (v - cy) * z / fy
    return np.array([x, y, z], dtype=float)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample_dir", default="data/realsense_sample")
    parser.add_argument("--extrinsic", default="config/camera_extrinsic.yaml")
    parser.add_argument("--out_manual", default="data/manual_grasp/manual_grasps.json")
    parser.add_argument("--out_grasp", default="data/graspnet_result/grasp_result.json")
    parser.add_argument("--z_offset", type=float, default=-0.015)
    parser.add_argument("--depth_radius", type=int, default=10)
    args = parser.parse_args()

    sample_dir = Path(args.sample_dir)
    color_path = sample_dir / "color.png"
    depth_path = sample_dir / "depth.png"
    info_path = sample_dir / "camera_info.json"

    color = cv2.imread(str(color_path), cv2.IMREAD_COLOR)
    depth = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)

    if color is None:
        raise SystemExit(f"Failed to read {color_path}")
    if depth is None:
        raise SystemExit(f"Failed to read {depth_path}")

    K = load_camera_info(info_path)

    cfg = yaml.safe_load(open(args.extrinsic))
    Rbc = np.array(cfg["T_base_camera"]["rotation_matrix"], dtype=float)
    tbc = np.array(cfg["T_base_camera"]["translation"], dtype=float)

    clicked = []
    vis = color.copy()

    print("操作说明：")
    print("  鼠标左键：按顺序点击抓取点")
    print("  u：撤销最后一个点")
    print("  s 或 Enter：保存")
    print("  q 或 Esc：退出不保存")
    print(f"  depth_radius = {args.depth_radius}, z_offset = {args.z_offset} m")

    def redraw():
        nonlocal vis
        vis = color.copy()
        for i, item in enumerate(clicked):
            u, v = item["pixel"]
            cv2.circle(vis, (u, v), 6, (0, 0, 255), -1)
            cv2.putText(
                vis,
                str(i),
                (u + 8, v - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 0, 255),
                2,
            )

    def on_mouse(event, x, y, flags, param):
        if event != cv2.EVENT_LBUTTONDOWN:
            return

        z = depth_at(depth, x, y, args.depth_radius)
        if z is None:
            print(f"[WARN] pixel=({x},{y}) has no valid depth")
            return

        p_cam = camera_point_from_pixel(x, y, z, K)
        p_base_raw = Rbc @ p_cam + tbc
        p_base = p_base_raw.copy()
        p_base[2] += args.z_offset

        item = {
            "index": len(clicked),
            "pixel": [int(x), int(y)],
            "depth_m": float(z),
            "camera_position": [float(a) for a in p_cam],
            "base_position_raw": [float(a) for a in p_base_raw],
            "base_position": [float(a) for a in p_base],
            "z_offset": float(args.z_offset),
        }

        clicked.append(item)
        redraw()

        print(
            f"click {item['index']}: pixel=({x},{y}), depth={z:.4f}, "
            f"base_raw=({p_base_raw[0]:.3f},{p_base_raw[1]:.3f},{p_base_raw[2]:.3f}), "
            f"base_final=({p_base[0]:.3f},{p_base[1]:.3f},{p_base[2]:.3f})"
        )

    cv2.namedWindow("manual_click_grasp_points", cv2.WINDOW_NORMAL)
    cv2.setMouseCallback("manual_click_grasp_points", on_mouse)

    while True:
        cv2.imshow("manual_click_grasp_points", vis)
        key = cv2.waitKey(30) & 0xFF

        if key in [ord("q"), 27]:
            print("quit without saving")
            cv2.destroyAllWindows()
            return

        if key == ord("u"):
            if clicked:
                removed = clicked.pop()
                print("undo:", removed["index"])
                for i, item in enumerate(clicked):
                    item["index"] = i
                redraw()

        if key in [ord("s"), 13, 10]:
            break

    cv2.destroyAllWindows()

    out_manual = Path(args.out_manual)
    out_manual.parent.mkdir(parents=True, exist_ok=True)

    data = {
        "frame_id": "base",
        "mode": "manual_click_fixed_orientation",
        "source_color": str(color_path),
        "source_depth": str(depth_path),
        "source_camera_info": str(info_path),
        "camera_extrinsic": str(args.extrinsic),
        "z_offset": float(args.z_offset),
        "depth_radius": int(args.depth_radius),
        "grasps": clicked,
        "note": "Manual clicked grasp points. Executor ignores orientation and uses fixed home orientation.",
    }

    json.dump(data, open(out_manual, "w"), indent=2)
    print("saved manual grasps:", out_manual)

    if not clicked:
        print("No clicked points, not writing grasp_result.json")
        return

    first = clicked[0]
    out = {
        "frame_id": "base",
        "position": first["base_position"],
        "orientation": [0.0, 0.0, 0.0, 1.0],
        "width": 0.075,
        "score": 1.0,
        "source": "manual_click",
        "manual_index": 0,
        "pixel": first["pixel"],
        "camera_position": first["camera_position"],
        "base_position_raw": first["base_position_raw"],
        "z_offset": float(args.z_offset),
        "note": "EXECUTION GRASP: manual click point. Orientation ignored by manual-click executor.",
    }

    out_grasp = Path(args.out_grasp)
    out_grasp.parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(out_grasp, "w"), indent=2)
    print("saved first grasp_result:", out_grasp)


if __name__ == "__main__":
    main()
