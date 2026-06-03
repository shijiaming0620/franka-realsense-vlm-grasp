#!/usr/bin/env python3
import argparse
import json
import math
import os
import signal
import sys
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
        arr = arr[:, :msg.width * 3].reshape(msg.height, msg.width, 3)
        if enc == "rgb8":
            arr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
        return arr.copy()

    if enc in ["16uc1", "mono16"]:
        arr = np.frombuffer(msg.data, dtype=np.uint16)
        arr = arr.reshape(msg.height, msg.step // 2)
        return arr[:, :msg.width].copy()

    if enc == "32fc1":
        arr = np.frombuffer(msg.data, dtype=np.float32)
        arr = arr.reshape(msg.height, msg.step // 4)
        return arr[:, :msg.width].copy()

    raise RuntimeError(f"Unsupported encoding: {msg.encoding}")


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
        d /= 1000.0
    return d


def camera_point_from_pixel(u, v, z, K):
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    return np.array([(u - cx) * z / fx, (v - cy) * z / fy, z], dtype=float)


def yaw_quat(yaw):
    return [0.0, 0.0, math.sin(yaw / 2.0), math.cos(yaw / 2.0)]


def wrap_angle(a):
    while a > math.pi:
        a -= 2 * math.pi
    while a < -math.pi:
        a += 2 * math.pi
    return a


class OnlineManualYaw(Node):
    def __init__(self, args):
        super().__init__("online_click_execute_manual_grasp_yaw")
        self.args = args

        cfg = yaml.safe_load(open(args.extrinsic))
        self.Rbc = np.array(cfg["T_base_camera"]["rotation_matrix"], dtype=float)
        self.tbc = np.array(cfg["T_base_camera"]["translation"], dtype=float)

        self.color = None
        self.depth = None
        self.K = None

        self.pending = None
        self.running = False
        self.bridge_proc = None
        self.executor_proc = None
        self.history = []
        self.vlm_overlay = None

        self.create_subscription(Image, args.color_topic, self.color_cb, 10)
        self.create_subscription(Image, args.depth_topic, self.depth_cb, 10)
        self.create_subscription(CameraInfo, args.camera_info_topic, self.info_cb, 10)

        Path(args.out_grasp).parent.mkdir(parents=True, exist_ok=True)
        Path(args.history_json).parent.mkdir(parents=True, exist_ok=True)

    def color_cb(self, msg):
        self.color = image_msg_to_cv2(msg)

    def depth_cb(self, msg):
        self.depth = image_msg_to_cv2(msg)

    def info_cb(self, msg):
        self.K = np.array(msg.k, dtype=float).reshape(3, 3)

    def point_from_pixel(self, u, v):
        if self.depth is None or self.K is None:
            return None
        z = depth_at(self.depth, u, v, self.args.depth_radius)
        if z is None:
            return None
        p_cam = camera_point_from_pixel(u, v, z, self.K)
        p_base_raw = self.Rbc @ p_cam + self.tbc
        p_base = p_base_raw.copy()
        p_base[2] += self.args.z_offset
        return z, p_cam, p_base_raw, p_base

    def select_center(self, u, v):
        result = self.point_from_pixel(u, v)
        if result is None:
            print(f"[WARN] pixel=({u},{v}) no valid depth")
            return

        z, p_cam, p_base_raw, p_base = result
        self.pending = {
            "index": len(self.history),
            "pixel": [int(u), int(v)],
            "depth_m": float(z),
            "camera_position": [float(x) for x in p_cam],
            "base_position_raw": [float(x) for x in p_base_raw],
            "base_position": [float(x) for x in p_base],
            "z_offset": float(self.args.z_offset),
            "yaw_rad": 0.0,
            "yaw_deg": 0.0,
            "direction_pixel": None,
        }

        print("\nSelected grasp center")
        print(f"  pixel=({u},{v}), depth={z:.4f}")
        print(f"  base_raw   = ({p_base_raw[0]:.3f}, {p_base_raw[1]:.3f}, {p_base_raw[2]:.3f})")
        print(f"  base_final = ({p_base[0]:.3f}, {p_base[1]:.3f}, {p_base[2]:.3f})")
        print("  yaw = 0 deg. Right-click direction or use a/d/z/c to rotate.")

    def set_yaw_by_direction(self, u, v):
        if self.pending is None:
            print("[WARN] select center first")
            return

        # 用抓取中心点的深度来反投影方向点。
        # 这样只利用图像上的方向，不依赖方向点本身的深度，避免点到桌面/边缘导致 yaw 抖动。
        center_depth = float(self.pending["depth_m"])
        p2_cam = camera_point_from_pixel(u, v, center_depth, self.K)
        p2_raw = self.Rbc @ p2_cam + self.tbc

        p1 = np.array(self.pending["base_position_raw"], dtype=float)
        d = p2_raw - p1

        if np.linalg.norm(d[:2]) < 1e-4:
            print("[WARN] direction too short")
            return

        yaw = wrap_angle(math.atan2(d[1], d[0]) + math.radians(self.args.yaw_offset_deg))
        self.pending["yaw_rad"] = float(yaw)
        self.pending["yaw_deg"] = float(yaw * 180.0 / math.pi)
        self.pending["direction_pixel"] = [int(u), int(v)]

        print(f"Set yaw by drawn image direction: {self.pending['yaw_deg']:.1f} deg")
        print("红线表示夹爪开口方向。长方体斜放时，让红线大致垂直跨过要夹的两侧。")

    def adjust_yaw(self, delta_deg):
        if self.pending is None:
            print("[WARN] no pending point")
            return
        yaw = wrap_angle(self.pending["yaw_rad"] + math.radians(delta_deg))
        self.pending["yaw_rad"] = float(yaw)
        self.pending["yaw_deg"] = float(yaw * 180.0 / math.pi)
        print(f"Adjusted yaw: {self.pending['yaw_deg']:.1f} deg")

    def write_grasp_json(self, item):
        out = {
            "frame_id": "base",
            "position": item["base_position"],
            "orientation": yaw_quat(item["yaw_rad"]),
            "width": self.args.width,
            "score": 1.0,
            "source": "online_manual_click_yaw",
            "manual_index": item["index"],
            "pixel": item["pixel"],
            "camera_position": item["camera_position"],
            "base_position_raw": item["base_position_raw"],
            "z_offset": item["z_offset"],
            "manual_yaw_rad": item["yaw_rad"],
            "manual_yaw_deg": item["yaw_deg"],
            "note": "Manual top-down grasp with yaw. Executor applies this yaw to fixed home orientation.",
        }
        json.dump(out, open(self.args.out_grasp, "w"), indent=2)

        data = {
            "mode": "online_manual_click_yaw",
            "grasps": self.history,
        }
        json.dump(data, open(self.args.history_json, "w"), indent=2)

        print("Saved grasp:", self.args.out_grasp)
        print(f"  yaw_deg = {item['yaw_deg']:.1f}")

    def kill_proc(self, p):
        if p is None:
            return
        try:
            os.killpg(os.getpgid(p.pid), signal.SIGTERM)
        except Exception:
            pass


    def run_vlm_selection(self):
        if self.running:
            print("[WARN] robot running")
            return
        if self.color is None:
            print("[WARN] no camera image yet")
            return

        image_path = Path(self.args.vlm_image_path)
        image_path.parent.mkdir(parents=True, exist_ok=True)

        # 这里保存的是当前实时窗口中的最新相机画面，不是旧照片
        cv2.imwrite(str(image_path), self.color)

        cmd = [
            sys.executable,
            "scripts/vlm_select_grasp_from_image.py",
            "--image", str(image_path),
            "--out_json", self.args.vlm_out_json,
            "--model", self.args.vlm_model,
            "--instruction", self.args.vlm_instruction,
        ]

        if self.args.vlm_crop_roi.strip():
            cmd += ["--crop_roi", self.args.vlm_crop_roi.strip()]

        print("\n========== VLM SELECT GRASP FROM CURRENT FRAME ==========")
        print("Saved current frame:", image_path)
        print("Running:", " ".join(cmd))

        try:
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError as e:
            print("[ERROR] VLM selection failed:", e)
            return

        try:
            data = json.load(open(self.args.vlm_out_json))
            data = self.refine_vlm_result_with_depth(data)
            with open(self.args.vlm_out_json, "w") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            self.vlm_overlay = data
            center = data["center_pixel"]
            direction = data["direction_pixel"]
        except Exception as e:
            print("[ERROR] failed to read VLM result:", e)
            return

        print("VLM target:", data.get("target_description", ""))
        print("VLM center_pixel:", center)
        print("VLM direction_pixel:", direction)
        print("VLM confidence:", data.get("confidence", None))
        print("VLM reason:", data.get("reason", ""))

        self.select_center(int(center[0]), int(center[1]))
        self.set_yaw_by_direction(int(direction[0]), int(direction[1]))

        print("========== VLM RESULT READY ==========")
        print("Check red point/line. Press e to execute, or adjust manually.\n")



    def _depth_meters(self):
        if self.depth is None:
            return None
        if self.depth.dtype == np.uint16:
            return self.depth.astype(np.float32) / 1000.0
        return self.depth.astype(np.float32)

    def _clamp_color_point(self, p):
        h, w = self.color.shape[:2]
        u = int(round(float(p[0])))
        v = int(round(float(p[1])))
        u = max(0, min(w - 1, u))
        v = max(0, min(h - 1, v))
        return [u, v]

    def _short_axis_direction_from_points(self, pts_color, center_color):
        pts = np.array(pts_color, dtype=np.float32)
        c = np.array(center_color, dtype=np.float32)

        if len(pts) < 3:
            return [int(c[0] + 70), int(c[1])]

        rect = cv2.minAreaRect(pts)
        box = cv2.boxPoints(rect).astype(np.float32)

        edges = []
        for i in range(4):
            p1 = box[i]
            p2 = box[(i + 1) % 4]
            d = p2 - p1
            length = float(np.linalg.norm(d))
            edges.append((length, d))

        edges.sort(key=lambda x: x[0])
        short_len, short_d = edges[0]
        long_len, _ = edges[-1]

        if long_len < 1e-6 or long_len / max(short_len, 1e-6) < 1.15:
            d = np.array([70.0, 0.0], dtype=np.float32)
        else:
            n = np.linalg.norm(short_d)
            d = np.array([70.0, 0.0], dtype=np.float32) if n < 1e-6 else short_d / n * 70.0

        q = c + d
        return self._clamp_color_point([q[0], q[1]])

    def refine_vlm_result_with_depth(self, data):
        """
        VLM 只做粗识别；这里用深度 + 外参 + base z 高度重新修正：
        1. object_bbox：用高于桌面的连通域修正整个物体区域
        2. top_surface_polygon：取该物体最高一层深度区域
        3. center_pixel：上表面中心
        4. direction_pixel：上表面短轴方向
        """
        if not getattr(self.args, "depth_refine", False):
            return data

        if self.color is None or self.depth is None or self.K is None:
            print("[WARN] geom refine skipped: color/depth/K not ready")
            return data

        if self.depth.dtype == np.uint16:
            D = self.depth.astype(np.float32) / 1000.0
        else:
            D = self.depth.astype(np.float32)

        color_h, color_w = self.color.shape[:2]
        depth_h, depth_w = D.shape[:2]

        sx = depth_w / float(color_w)
        sy = depth_h / float(color_h)

        fx = float(self.K[0, 0])
        fy = float(self.K[1, 1])
        cx = float(self.K[0, 2])
        cy = float(self.K[1, 2])

        R = self.Rbc
        t = self.tbc

        table_z = float(self.args.geom_table_z)
        min_height = float(self.args.geom_min_height)
        top_band = float(self.args.geom_top_band)
        pad = int(self.args.geom_pad)

        def clamp_point(p):
            u = int(round(float(p[0])))
            v = int(round(float(p[1])))
            u = max(0, min(color_w - 1, u))
            v = max(0, min(color_h - 1, v))
            return [u, v]

        def clamp_bbox(b):
            x1, y1, x2, y2 = [int(round(float(x))) for x in b]
            x1 = max(0, min(color_w - 1, x1))
            y1 = max(0, min(color_h - 1, y1))
            x2 = max(0, min(color_w - 1, x2))
            y2 = max(0, min(color_h - 1, y2))
            if x2 < x1:
                x1, x2 = x2, x1
            if y2 < y1:
                y1, y2 = y2, y1
            return [x1, y1, x2, y2]

        def short_axis_direction(points, center):
            pts = np.array(points, dtype=np.float32)
            c = np.array(center, dtype=np.float32)

            if len(pts) < 3:
                return clamp_point([c[0] + 70, c[1]])

            rect = cv2.minAreaRect(pts)
            box = cv2.boxPoints(rect).astype(np.float32)

            edges = []
            for i in range(4):
                p1 = box[i]
                p2 = box[(i + 1) % 4]
                d = p2 - p1
                length = float(np.linalg.norm(d))
                edges.append((length, d))

            edges.sort(key=lambda x: x[0])
            short_len, short_d = edges[0]
            long_len, _ = edges[-1]

            if long_len < 1e-6 or long_len / max(short_len, 1e-6) < 1.15:
                d = np.array([70.0, 0.0], dtype=np.float32)
            else:
                n = np.linalg.norm(short_d)
                d = np.array([70.0, 0.0], dtype=np.float32) if n < 1e-6 else short_d / n * 70.0

            q = c + d
            return clamp_point([q[0], q[1]])

        def refine_one_object(obj, idx):
            bbox = obj.get("object_bbox")
            if not bbox or len(bbox) != 4:
                return obj, False

            x1, y1, x2, y2 = clamp_bbox(bbox)
            x1 = max(0, x1 - pad)
            y1 = max(0, y1 - pad)
            x2 = min(color_w - 1, x2 + pad)
            y2 = min(color_h - 1, y2 + pad)

            if x2 <= x1 or y2 <= y1:
                return obj, False

            # ROI 中每个 color 像素对应一个 depth 像素
            us = np.arange(x1, x2 + 1)
            vs = np.arange(y1, y2 + 1)
            Uc, Vc = np.meshgrid(us, vs)

            Ud = np.clip(np.round(Uc * sx).astype(np.int32), 0, depth_w - 1)
            Vd = np.clip(np.round(Vc * sy).astype(np.int32), 0, depth_h - 1)

            Z = D[Vd, Ud]
            valid = np.isfinite(Z) & (Z > 0.05) & (Z < 3.0)

            if valid.sum() < 50:
                return obj, False

            Xc = (Uc.astype(np.float32) - cx) * Z / fx
            Yc = (Vc.astype(np.float32) - cy) * Z / fy

            # 只需要 base_z
            base_z = (
                R[2, 0] * Xc +
                R[2, 1] * Yc +
                R[2, 2] * Z +
                t[2]
            )

            # 高于桌面的区域认为是物体
            obj_mask = valid & (base_z > table_z + min_height)

            mask_u8 = (obj_mask.astype(np.uint8) * 255)
            kernel = np.ones((5, 5), np.uint8)
            mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_OPEN, kernel)
            mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_CLOSE, kernel)

            num, labels, stats, centroids = cv2.connectedComponentsWithStats(mask_u8, 8)
            if num <= 1:
                return obj, False

            # 选靠近 VLM 中心的连通域
            center = obj.get("center_pixel", data.get("center_pixel", [(x1 + x2)//2, (y1 + y2)//2]))
            cu, cv = clamp_point(center)
            cx_roi = cu - x1
            cy_roi = cv - y1

            best_label = None
            best_score = -1.0

            for lab in range(1, num):
                area = stats[lab, cv2.CC_STAT_AREA]
                if area < 80:
                    continue

                lx, ly = centroids[lab]
                dist = np.hypot(lx - cx_roi, ly - cy_roi)
                score = float(area) / (1.0 + dist / 80.0)

                if 0 <= cx_roi < labels.shape[1] and 0 <= cy_roi < labels.shape[0]:
                    if labels[int(cy_roi), int(cx_roi)] == lab:
                        score *= 2.0

                if score > best_score:
                    best_score = score
                    best_label = lab

            if best_label is None:
                return obj, False

            comp = labels == best_label

            # 整个物体框：高于桌面的连通域
            comp_u8 = (comp.astype(np.uint8) * 255)
            contours, _ = cv2.findContours(comp_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not contours:
                return obj, False

            cnt = max(contours, key=cv2.contourArea)
            bx, by, bw, bh = cv2.boundingRect(cnt)
            object_bbox = [x1 + bx, y1 + by, x1 + bx + bw, y1 + by + bh]

            # 上表面：该连通域中 base_z 最高的一层
            obj_z_vals = base_z[comp & valid]
            if obj_z_vals.size < 30:
                return obj, False

            z_top = float(np.percentile(obj_z_vals, 90))
            top_mask = comp & valid & (base_z >= z_top - top_band)

            top_u8 = (top_mask.astype(np.uint8) * 255)
            top_u8 = cv2.morphologyEx(top_u8, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
            top_u8 = cv2.morphologyEx(top_u8, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))

            top_contours, _ = cv2.findContours(top_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not top_contours:
                return obj, False

            top_cnt = max(top_contours, key=cv2.contourArea)
            if cv2.contourArea(top_cnt) < 20:
                return obj, False

            # 上表面最小外接矩形
            pts_roi = top_cnt.reshape(-1, 2).astype(np.float32)
            pts_color = pts_roi.copy()
            pts_color[:, 0] += x1
            pts_color[:, 1] += y1

            rect = cv2.minAreaRect(pts_color)
            box = cv2.boxPoints(rect)
            top_poly = [[int(round(p[0])), int(round(p[1]))] for p in box]
            top_poly = [clamp_point(p) for p in top_poly]

            # 上表面中心
            M = cv2.moments(top_cnt)
            if abs(M["m00"]) > 1e-6:
                center_color = [
                    int(round(x1 + M["m10"] / M["m00"])),
                    int(round(y1 + M["m01"] / M["m00"]))
                ]
            else:
                center_color = [
                    int(round(np.mean([p[0] for p in top_poly]))),
                    int(round(np.mean([p[1] for p in top_poly])))
                ]

            center_color = clamp_point(center_color)
            direction_color = short_axis_direction(top_poly, center_color)

            obj["object_bbox"] = clamp_bbox(object_bbox)
            obj["top_surface_polygon"] = top_poly
            obj["center_pixel"] = center_color
            obj["direction_pixel"] = direction_color
            obj["geom_refined"] = True
            obj["geom_top_z"] = z_top
            obj["geom_table_z"] = table_z
            obj["geom_object_area_px"] = int(cv2.contourArea(cnt))
            obj["geom_top_area_px"] = int(cv2.contourArea(top_cnt))

            print(
                f"[INFO] geom refine object {idx}: "
                f"name={obj.get('object_name_en', '')}, "
                f"bbox={obj['object_bbox']}, "
                f"top={top_poly}, "
                f"center={center_color}, "
                f"top_z={z_top:.3f}"
            )

            return obj, True

        objects = data.get("objects", [])
        refined_count = 0

        if objects:
            new_objects = []
            for i, obj in enumerate(objects):
                new_obj, ok = refine_one_object(obj, i)
                if ok:
                    refined_count += 1
                new_objects.append(new_obj)

            data["objects"] = new_objects

            selected_index = int(data.get("selected_index", 0))
            selected_index = max(0, min(len(new_objects) - 1, selected_index))
            selected = new_objects[selected_index]
        else:
            selected, ok = refine_one_object(data, 0)
            refined_count = 1 if ok else 0

        if refined_count <= 0:
            print("[WARN] geom refine failed for all objects; keeping VLM result")
            return data

        data["target_description"] = selected.get("target_description", "")
        data["object_name_en"] = selected.get("object_name_en", "")
        data["object_bbox"] = selected.get("object_bbox")
        data["top_surface_polygon"] = selected.get("top_surface_polygon")
        data["center_pixel"] = selected.get("center_pixel")
        data["direction_pixel"] = selected.get("direction_pixel")
        data["geom_refined"] = True
        data["geom_refined_count"] = refined_count

        return data


    def draw_vlm_overlay(self, canvas):
        data = self.vlm_overlay
        if not data:
            return canvas

        objects = data.get("objects", [])
        selected_index = int(data.get("selected_index", -1))

        for i, obj in enumerate(objects):
            selected = (i == selected_index)

            bbox_color = (0, 255, 0) if selected else (255, 255, 0)
            poly_color = (255, 0, 255) if selected else (180, 0, 180)

            bbox = obj.get("object_bbox")
            if bbox and len(bbox) == 4:
                x1, y1, x2, y2 = [int(v) for v in bbox]
                cv2.rectangle(canvas, (x1, y1), (x2, y2), bbox_color, 3 if selected else 2)

                name = obj.get("object_name_en", f"object_{i}")
                label = f"{i} {name} {float(obj.get('auto_score', 0.0)):.2f}"
                if selected:
                    label = "SELECTED " + label

                cv2.putText(
                    canvas,
                    label,
                    (x1, max(25, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    bbox_color,
                    2,
                )

            poly = obj.get("top_surface_polygon")
            if poly and len(poly) >= 3:
                pts = np.array(poly, dtype=np.int32)
                cv2.polylines(canvas, [pts], isClosed=True, color=poly_color, thickness=3 if selected else 2)
                for pnt in poly:
                    cv2.circle(canvas, tuple([int(pnt[0]), int(pnt[1])]), 4, poly_color, -1)

        return canvas

    def execute_pending(self):
        if self.running:
            print("[WARN] robot running")
            return
        if self.pending is None:
            print("[WARN] no selected point")
            return

        item = self.pending
        item["index"] = len(self.history)
        self.history.append(item)
        self.pending = None
        self.write_grasp_json(item)

        threading.Thread(target=self.execute_once, daemon=True).start()

    def execute_once(self):
        self.running = True
        grasp_json = str(Path(self.args.out_grasp).resolve())

        bridge_cmd = [
            "ros2", "run", "franka_grasp_demo", "graspnet_bridge_node.py",
            "--ros-args",
            "-p", f"json_path:={grasp_json}",
            "-p", "publish_once:=false",
        ]
        executor_cmd = ["ros2", "launch", "franka_grasp_demo", "executor.launch.py"]

        print("\n========== START EXECUTION ==========")
        self.bridge_proc = subprocess.Popen(bridge_cmd, preexec_fn=os.setsid)
        time.sleep(2.0)

        self.executor_proc = subprocess.Popen(executor_cmd, preexec_fn=os.setsid)
        ret = self.executor_proc.wait()

        print(f"Executor finished, return code={ret}")
        self.kill_proc(self.bridge_proc)
        self.bridge_proc = None
        self.executor_proc = None
        self.running = False
        print("========== READY FOR NEXT CLICK ==========\n")

    def shutdown_processes(self):
        self.kill_proc(self.executor_proc)
        self.kill_proc(self.bridge_proc)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--color_topic", default="/camera/camera/color/image_raw")
    parser.add_argument("--depth_topic", default="/camera/camera/aligned_depth_to_color/image_raw")
    parser.add_argument("--camera_info_topic", default="/camera/camera/color/camera_info")
    parser.add_argument("--extrinsic", default="config/camera_extrinsic.yaml")
    parser.add_argument("--out_grasp", default="data/graspnet_result/grasp_result.json")
    parser.add_argument("--history_json", default="data/manual_grasp/online_manual_yaw_history.json")
    parser.add_argument("--z_offset", type=float, default=-0.030)
    parser.add_argument("--depth_radius", type=int, default=10)
    parser.add_argument("--width", type=float, default=0.075)
    parser.add_argument("--vlm_crop_roi", default="")
    parser.add_argument("--vlm_model", default="qwen-vl-plus")
    parser.add_argument("--vlm_image_path", default="data/manual_grasp/vlm_current_frame.png")
    parser.add_argument("--vlm_out_json", default="data/manual_grasp/vlm_result.json")
    parser.add_argument(
        "--vlm_instruction",
        default="抓取画面中最适合从上方二指夹取的长方体或积木。选择物体上表面中心，方向线表示夹爪开口方向。"
    )
    parser.add_argument("--yaw_offset_deg", type=float, default=0.0)
    parser.add_argument("--depth_refine", action="store_true")
    parser.add_argument("--depth_refine_thresh", type=float, default=0.045)
    parser.add_argument("--depth_refine_min_area", type=int, default=80)
    parser.add_argument("--geom_table_z", type=float, default=-0.080)
    parser.add_argument("--geom_min_height", type=float, default=0.025)
    parser.add_argument("--geom_top_band", type=float, default=0.025)
    parser.add_argument("--geom_pad", type=int, default=20)
    args = parser.parse_args()

    rclpy.init()
    node = OnlineManualYaw(args)

    print("\n在线点选 + yaw 抓取说明：")
    print("  左键：选抓取中心")
    print("  右键：选夹爪开口方向")
    print("  a/d：yaw -2.5/+2.5 deg")
    print("  z/c：yaw -1/+1 deg")
    print("  u：取消当前点")
    print("  g：大模型自动推荐抓取中心和方向")
    print("  e：执行当前点")
    print("  q/Esc：退出")
    print("  红线表示夹爪开口方向。")
    print(f"  yaw_offset_deg = {args.yaw_offset_deg}")
    print()

    win = "online_click_execute_manual_grasp_yaw"

    def mouse_cb(event, x, y, flags, param):
        if node.running:
            print("[WARN] robot running")
            return
        if event == cv2.EVENT_LBUTTONDOWN:
            node.select_center(x, y)
        elif event == cv2.EVENT_RBUTTONDOWN:
            node.set_yaw_by_direction(x, y)

    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(win, mouse_cb)

    try:
        while rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0.01)

            if node.color is None:
                canvas = np.zeros((480, 640, 3), dtype=np.uint8)
                cv2.putText(canvas, "Waiting for camera...", (40, 240),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
            else:
                canvas = node.color.copy()

            if hasattr(node, "draw_vlm_overlay"):
                canvas = node.draw_vlm_overlay(canvas)

            if node.pending is not None:
                u, v = node.pending["pixel"]
                yaw = node.pending["yaw_rad"]
                direction_pixel = node.pending.get("direction_pixel")

                if direction_pixel is not None:
                    x2, y2 = direction_pixel
                else:
                    length = 70
                    x2 = int(u + length * math.cos(yaw))
                    y2 = int(v + length * math.sin(yaw))

                cv2.circle(canvas, (u, v), 8, (0, 0, 255), -1)
                cv2.line(canvas, (u, v), (x2, y2), (0, 0, 255), 3)
                cv2.circle(canvas, (x2, y2), 5, (0, 0, 255), -1)
                cv2.putText(canvas, f"yaw={node.pending['yaw_deg']:.1f} deg",
                            (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

            if node.running:
                cv2.putText(canvas, "ROBOT RUNNING...", (20, 70),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
            else:
                cv2.putText(canvas, "Left click center, right click direction, e execute",
                            (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)

            cv2.imshow(win, canvas)
            key = cv2.waitKey(30) & 0xFF

            if key in [ord("q"), 27]:
                break
            elif key == ord("u"):
                node.pending = None
                node.vlm_overlay = None
                print("Cleared pending point and VLM overlay")
            elif key == ord("a"):
                node.adjust_yaw(-2.5)
            elif key == ord("d"):
                node.adjust_yaw(2.5)
            elif key == ord("z"):
                node.adjust_yaw(-1.0)
            elif key == ord("c"):
                node.adjust_yaw(1.0)
            elif key == ord("g"):
                node.run_vlm_selection()
            elif key == ord("e"):
                node.execute_pending()

    finally:
        node.shutdown_processes()
        cv2.destroyAllWindows()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
