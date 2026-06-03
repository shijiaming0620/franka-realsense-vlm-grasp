#!/usr/bin/env python3
import argparse
import json
import os
import signal
import subprocess
import threading
import time
from pathlib import Path

import cv2
import numpy as np
import yaml

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo


def image_msg_to_cv2(msg):
    enc = msg.encoding.lower()

    if enc in ["bgr8", "rgb8"]:
        arr = np.frombuffer(msg.data, dtype=np.uint8)
        arr = arr.reshape(msg.height, msg.step)
        arr = arr[:, :msg.width * 3]
        arr = arr.reshape(msg.height, msg.width, 3)
        if enc == "rgb8":
            arr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
        return arr.copy()

    if enc in ["mono8"]:
        arr = np.frombuffer(msg.data, dtype=np.uint8)
        arr = arr.reshape(msg.height, msg.step)
        arr = arr[:, :msg.width]
        return arr.copy()

    if enc in ["16uc1", "mono16"]:
        arr = np.frombuffer(msg.data, dtype=np.uint16)
        arr = arr.reshape(msg.height, msg.step // 2)
        arr = arr[:, :msg.width]
        return arr.copy()

    if enc in ["32fc1"]:
        arr = np.frombuffer(msg.data, dtype=np.float32)
        arr = arr.reshape(msg.height, msg.step // 4)
        arr = arr[:, :msg.width]
        return arr.copy()

    raise RuntimeError(f"Unsupported image encoding: {msg.encoding}")


def depth_at(depth, u, v, radius):
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


class OnlineClickExecutor(Node):
    def __init__(self, args):
        super().__init__("online_click_execute_manual_grasp")

        self.args = args

        cfg = yaml.safe_load(open(args.extrinsic))
        self.Rbc = np.array(cfg["T_base_camera"]["rotation_matrix"], dtype=float)
        self.tbc = np.array(cfg["T_base_camera"]["translation"], dtype=float)

        self.color = None
        self.depth = None
        self.K = None

        self.pending_item = None
        self.history = []
        self.running = False
        self.bridge_proc = None
        self.executor_proc = None

        self.create_subscription(Image, args.color_topic, self.color_cb, 10)
        self.create_subscription(Image, args.depth_topic, self.depth_cb, 10)
        self.create_subscription(CameraInfo, args.camera_info_topic, self.info_cb, 10)

        Path(args.out_grasp).parent.mkdir(parents=True, exist_ok=True)
        Path(args.history_json).parent.mkdir(parents=True, exist_ok=True)

    def color_cb(self, msg):
        try:
            self.color = image_msg_to_cv2(msg)
        except Exception as e:
            self.get_logger().error(f"color convert failed: {e}")

    def depth_cb(self, msg):
        try:
            self.depth = image_msg_to_cv2(msg)
        except Exception as e:
            self.get_logger().error(f"depth convert failed: {e}")

    def info_cb(self, msg):
        self.K = np.array(msg.k, dtype=float).reshape(3, 3)

    def make_item(self, u, v):
        if self.color is None or self.depth is None or self.K is None:
            print("[WARN] camera data not ready yet.")
            return None

        z = depth_at(self.depth, u, v, self.args.depth_radius)
        if z is None:
            print(f"[WARN] pixel=({u},{v}) has no valid depth.")
            return None

        p_cam = camera_point_from_pixel(u, v, z, self.K)
        p_base_raw = self.Rbc @ p_cam + self.tbc
        p_base = p_base_raw.copy()
        p_base[2] += self.args.z_offset

        item = {
            "index": len(self.history),
            "pixel": [int(u), int(v)],
            "depth_m": float(z),
            "camera_position": [float(x) for x in p_cam],
            "base_position_raw": [float(x) for x in p_base_raw],
            "base_position": [float(x) for x in p_base],
            "z_offset": float(self.args.z_offset),
            "depth_radius": int(self.args.depth_radius),
        }

        print(
            f"\nSelected point index={item['index']}: "
            f"pixel=({u},{v}), depth={z:.4f} m\n"
            f"  base_raw   = ({p_base_raw[0]:.3f}, {p_base_raw[1]:.3f}, {p_base_raw[2]:.3f})\n"
            f"  base_final = ({p_base[0]:.3f}, {p_base[1]:.3f}, {p_base[2]:.3f})"
        )

        return item

    def write_grasp_json(self, item):
        out = {
            "frame_id": "base",
            "position": item["base_position"],
            "orientation": [0.0, 0.0, 0.0, 1.0],
            "width": self.args.width,
            "score": 1.0,
            "source": "online_manual_click",
            "manual_index": item["index"],
            "pixel": item["pixel"],
            "camera_position": item["camera_position"],
            "base_position_raw": item["base_position_raw"],
            "z_offset": item["z_offset"],
            "note": "EXECUTION GRASP: online manual click. Orientation ignored by manual-click executor.",
        }

        json.dump(out, open(self.args.out_grasp, "w"), indent=2)
        print("Saved grasp json:", self.args.out_grasp)

        data = {
            "frame_id": "base",
            "mode": "online_manual_click",
            "z_offset": float(self.args.z_offset),
            "depth_radius": int(self.args.depth_radius),
            "grasps": self.history,
        }
        json.dump(data, open(self.args.history_json, "w"), indent=2)
        print("Saved history:", self.args.history_json)

    def start_execution_thread(self):
        if self.running:
            print("[WARN] Robot is already running. Wait until current execution finishes.")
            return

        if self.pending_item is None:
            print("[WARN] No selected point. Left-click a grasp point first.")
            return

        item = self.pending_item
        self.pending_item = None
        self.history.append(item)
        item["index"] = len(self.history) - 1
        self.write_grasp_json(item)

        th = threading.Thread(target=self.execute_once, daemon=True)
        th.start()

    def kill_process(self, proc):
        if proc is None:
            return
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        except Exception:
            pass

    def execute_once(self):
        self.running = True

        grasp_json_abs = str(Path(self.args.out_grasp).resolve())

        bridge_cmd = [
            "ros2", "run", "franka_grasp_demo", "graspnet_bridge_node.py",
            "--ros-args",
            "-p", f"json_path:={grasp_json_abs}",
            "-p", "publish_once:=false",
        ]

        executor_cmd = [
            "ros2", "launch", "franka_grasp_demo", "executor.launch.py",
        ]

        print("\n========== START EXECUTION ==========")
        print("Starting bridge...")
        self.bridge_proc = subprocess.Popen(
            bridge_cmd,
            preexec_fn=os.setsid,
        )

        time.sleep(2.0)

        print("Starting executor...")
        self.executor_proc = subprocess.Popen(
            executor_cmd,
            preexec_fn=os.setsid,
        )

        ret = self.executor_proc.wait()
        print(f"Executor finished with return code: {ret}")

        print("Stopping bridge...")
        self.kill_process(self.bridge_proc)
        self.bridge_proc = None
        self.executor_proc = None

        self.running = False
        print("========== READY FOR NEXT CLICK ==========\n")

    def shutdown_processes(self):
        self.kill_process(self.executor_proc)
        self.kill_process(self.bridge_proc)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--color_topic", default="/camera/camera/color/image_raw")
    parser.add_argument("--depth_topic", default="/camera/camera/aligned_depth_to_color/image_raw")
    parser.add_argument("--camera_info_topic", default="/camera/camera/color/camera_info")
    parser.add_argument("--extrinsic", default="config/camera_extrinsic.yaml")
    parser.add_argument("--out_grasp", default="data/graspnet_result/grasp_result.json")
    parser.add_argument("--history_json", default="data/manual_grasp/online_manual_history.json")
    parser.add_argument("--z_offset", type=float, default=-0.030)
    parser.add_argument("--depth_radius", type=int, default=10)
    parser.add_argument("--width", type=float, default=0.075)
    args = parser.parse_args()

    rclpy.init()
    node = OnlineClickExecutor(args)

    window = "online_click_execute_manual_grasp"

    print("\n在线点选抓取说明：")
    print("  左键：选择当前抓取点")
    print("  e：执行当前选中的点")
    print("  u：取消当前选中点")
    print("  q 或 Esc：退出")
    print(f"  z_offset = {args.z_offset} m")
    print(f"  depth_radius = {args.depth_radius}")
    print("  注意：执行过程中不要继续点击，等提示 READY FOR NEXT CLICK。\n")

    def on_mouse(event, x, y, flags, param):
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        if node.running:
            print("[WARN] Robot is running. Wait before selecting next point.")
            return
        item = node.make_item(x, y)
        if item is not None:
            node.pending_item = item

    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(window, on_mouse)

    try:
        while rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0.01)

            if node.color is None:
                canvas = np.zeros((480, 640, 3), dtype=np.uint8)
                cv2.putText(canvas, "Waiting for camera image...", (40, 240),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
            else:
                canvas = node.color.copy()

            if node.pending_item is not None:
                u, v = node.pending_item["pixel"]
                cv2.circle(canvas, (u, v), 8, (0, 0, 255), -1)
                cv2.putText(canvas, "PENDING - press e to execute", (20, 35),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

            if node.running:
                cv2.putText(canvas, "ROBOT RUNNING...", (20, 70),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
            else:
                cv2.putText(canvas, "Click point, press e to execute", (20, 70),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            cv2.imshow(window, canvas)
            key = cv2.waitKey(30) & 0xFF

            if key in [ord("q"), 27]:
                break

            if key == ord("u"):
                node.pending_item = None
                print("Cleared pending point.")

            if key == ord("e"):
                node.start_execution_thread()

    finally:
        node.shutdown_processes()
        cv2.destroyAllWindows()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
